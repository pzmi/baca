"""JSON-RPC 2.0 stdio client for the Usiec Cepra engine.

The engine server speaks Content-Length framed JSON-RPC 2.0 over stdin/stdout
(see ``slay-the-ceper/src/rpc/JsonRpcServer.js``). This module spawns the server
as a subprocess and exposes a synchronous ``call`` method plus a queue of
server-pushed notifications.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class RpcError(Exception):
    """Raised when the JSON-RPC server returns an error response."""

    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class RpcTransportError(Exception):
    """Raised when the subprocess dies or the stream closes unexpectedly."""


class RpcClient:
    """Synchronous JSON-RPC 2.0 client over a Node subprocess.

    One client instance owns one subprocess. Multiple runs are addressed via
    ``runId`` parameters on the engine methods — up to 16 concurrent runs per
    process (``RUN_CAP`` in the engine).
    """

    def __init__(
        self,
        engine_dir: str | Path | None = None,
        node_bin: str = "node",
        script: str = "scripts/rpc-server.js",
    ) -> None:
        self._engine_dir = Path(
            engine_dir
            if engine_dir is not None
            else os.environ.get("BACA_ENGINE_DIR", "../slay-the-ceper")
        ).resolve()
        if not (self._engine_dir / script).is_file():
            raise FileNotFoundError(
                f"RPC server script not found at {self._engine_dir / script}"
            )

        self._proc = subprocess.Popen(
            [node_bin, script],
            cwd=str(self._engine_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self._next_id = 1
        self._id_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._closed = threading.Event()

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> Any:  # noqa: ANN401 — RPC result is genuinely untyped
        if self._closed.is_set():
            raise RpcTransportError("RPC client is closed")

        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1

        resp_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[req_id] = resp_queue

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        self._send(request)

        try:
            response = resp_queue.get(timeout=timeout)
        except queue.Empty as e:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise RpcTransportError(
                f"Timeout waiting for response to {method}"
            ) from e

        if "error" in response:
            err = response["error"]
            raise RpcError(err["code"], err["message"], err.get("data"))
        return response.get("result")

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return all queued server-pushed notifications and clear the queue."""
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._notifications.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)

    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise RpcTransportError("Subprocess stdin is not available")
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            self._proc.stdin.write(header + body)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RpcTransportError("Subprocess stdin closed") from e

    def _reader_loop(self) -> None:
        stdout = self._proc.stdout
        assert stdout is not None
        buf = bytearray()
        while not self._closed.is_set():
            chunk = stdout.read(4096)
            if not chunk:
                self._closed.set()
                self._fail_all_pending(RpcTransportError("Subprocess stdout closed"))
                return
            buf.extend(chunk)
            while True:
                sep = buf.find(b"\r\n\r\n")
                if sep == -1:
                    break
                header = bytes(buf[:sep]).decode("ascii", errors="replace")
                content_length = self._parse_content_length(header)
                if content_length is None:
                    del buf[: sep + 4]
                    continue
                body_start = sep + 4
                if len(buf) < body_start + content_length:
                    break
                body = bytes(buf[body_start : body_start + content_length])
                del buf[: body_start + content_length]
                try:
                    msg = json.loads(body)
                except json.JSONDecodeError:
                    continue
                self._dispatch(msg)

    @staticmethod
    def _parse_content_length(header: str) -> int | None:
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                _, _, value = line.partition(":")
                try:
                    return int(value.strip())
                except ValueError:
                    return None
        return None

    def _dispatch(self, msg: dict[str, Any]) -> None:
        if "id" in msg and msg["id"] is not None and "method" not in msg:
            req_id = msg["id"]
            if not isinstance(req_id, int):
                return
            with self._pending_lock:
                q = self._pending.pop(req_id, None)
            if q is not None:
                q.put(msg)
            return
        if "method" in msg:
            self._notifications.put(msg)

    def _stderr_loop(self) -> None:
        stderr = self._proc.stderr
        assert stderr is not None
        while not self._closed.is_set():
            line = stderr.readline()
            if not line:
                return

    def _fail_all_pending(self, error: Exception) -> None:
        with self._pending_lock:
            pending = self._pending
            self._pending = {}
        for q in pending.values():
            q.put(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": str(error)},
                }
            )
