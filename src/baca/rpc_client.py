"""JSON-RPC 2.0 stdio client for the Usiec Cepra engine.

The engine server speaks Content-Length framed JSON-RPC 2.0 over stdin/stdout
(see ``slay-the-ceper/src/rpc/JsonRpcServer.js``). This module spawns the server
as a subprocess and exposes a synchronous ``call`` method plus a queue of
server-pushed notifications.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import orjson

logger = logging.getLogger(__name__)

DEFAULT_CALL_TIMEOUT = 30.0


class RpcError(Exception):
    """Raised when the JSON-RPC server returns an error response."""

    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class RpcTransportError(Exception):
    """Raised when the subprocess dies or the stream closes unexpectedly."""


class RpcTimeoutError(RpcTransportError):
    """Raised when a JSON-RPC response does not arrive within the call timeout."""

    def __init__(self, method: str, req_id: int, timeout: float) -> None:
        super().__init__(
            f"Timeout after {timeout:.1f}s waiting for response to {method!r} (id={req_id})"
        )
        self.method = method
        self.req_id = req_id
        self.timeout = timeout


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
        default_timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> None:
        self._engine_dir = Path(
            engine_dir
            if engine_dir is not None
            else os.environ.get("BACA_ENGINE_DIR", "../slay-the-ceper")
        ).resolve()
        if not (self._engine_dir / script).is_file():
            raise FileNotFoundError(f"RPC server script not found at {self._engine_dir / script}")

        self._default_timeout = default_timeout
        self._proc = subprocess.Popen(  # noqa: S603 — node binary + engine script under our control
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
        self._write_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._fatal_lock = threading.Lock()

        self._reader = threading.Thread(target=self._reader_loop, name="rpc-reader", daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._stderr_loop, name="rpc-stderr", daemon=True
        )
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
        timeout: float | None = None,
    ) -> Any:
        effective_timeout = self._default_timeout if timeout is None else timeout

        self._raise_if_fatal()
        if self._closed.is_set():
            raise RpcTransportError("RPC client is closed")
        self._raise_if_subprocess_dead()

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

        try:
            self._send(request)
        except BaseException:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise

        try:
            response = resp_queue.get(timeout=effective_timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            self._raise_if_fatal()
            self._raise_if_subprocess_dead()
            raise RpcTimeoutError(method, req_id, effective_timeout) from None

        if response.get("__baca_fatal__"):
            self._raise_if_fatal()
            raise RpcTransportError("RPC client entered failed state while awaiting response")

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
            with self._write_lock, contextlib.suppress(BrokenPipeError, OSError):
                self._proc.stdin.close()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)
        self._fail_all_pending(RpcTransportError("RPC client closed"))

    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise RpcTransportError("Subprocess stdin is not available")
        body = orjson.dumps(msg)
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        payload = header + body
        try:
            with self._write_lock:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._set_fatal(RpcTransportError(f"Subprocess stdin closed: {e}"))
            raise RpcTransportError("Subprocess stdin closed") from e

    def _reader_loop(self) -> None:
        try:
            self._run_reader()
        except BaseException as e:
            logger.exception("rpc-reader thread crashed: %s", e)
            self._set_fatal(e)
            self._fail_all_pending(e)
            return
        # Normal EOF exit — already handled in _run_reader.

    def _run_reader(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            raise RpcTransportError("Subprocess stdout is not available")
        buf = bytearray()
        while not self._closed.is_set():
            chunk = stdout.read(4096)
            if not chunk:
                return_code = self._proc.poll()
                err = RpcTransportError(f"Subprocess stdout closed (returncode={return_code})")
                logger.warning("rpc-reader: EOF on subprocess stdout (returncode=%s)", return_code)
                self._set_fatal(err)
                self._fail_all_pending(err)
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
                    msg = orjson.loads(body)
                except orjson.JSONDecodeError:
                    logger.warning("rpc-reader: dropping malformed JSON body (%d bytes)", len(body))
                    continue
                try:
                    self._dispatch(msg)
                except Exception:
                    logger.exception(
                        "rpc-reader: dispatch failed for message id=%r; continuing",
                        msg.get("id") if isinstance(msg, dict) else None,
                    )

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
        if stderr is None:
            return
        while not self._closed.is_set():
            try:
                line = stderr.readline()
            except ValueError, OSError:
                return
            if not line:
                return
            logger.debug("rpc-stderr: %s", line.decode("utf-8", errors="replace").rstrip())

    def _fail_all_pending(self, error: BaseException) -> None:
        with self._pending_lock:
            pending = self._pending
            self._pending = {}
        sentinel = {"__baca_fatal__": True, "error_repr": repr(error)}
        for q in pending.values():
            with contextlib.suppress(queue.Full):
                q.put_nowait(sentinel)

    def _set_fatal(self, error: BaseException) -> None:
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = error

    def _raise_if_fatal(self) -> None:
        with self._fatal_lock:
            err = self._fatal_error
        if err is not None:
            raise RpcTransportError(f"RPC client in failed state: {err}") from err

    def _raise_if_subprocess_dead(self) -> None:
        rc = self._proc.poll()
        if rc is not None:
            err = RpcTransportError(f"Subprocess exited with returncode={rc}")
            self._set_fatal(err)
            self._fail_all_pending(err)
            raise err
