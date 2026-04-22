"""Python wrapper around the HeuristicBot Node subprocess.

Speaks line-delimited JSON over the subprocess's stdin/stdout. One observation
in, one action out. Diagnostics from the Node side arrive on stderr; any bytes
on stderr poison the bridge (we raise on the next ``decide`` call).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self

logger = logging.getLogger(__name__)

DEFAULT_DECIDE_TIMEOUT = 5.0
_RUNNER_SCRIPT = Path(__file__).resolve().parent / "heuristic_runner.js"


class HeuristicBridge:
    """Spawn a persistent Node process that evaluates HeuristicBot on demand.

    One bridge owns one subprocess. ``decide`` is synchronous and serialized;
    this class is not thread-safe for concurrent ``decide`` calls.
    """

    def __init__(
        self,
        engine_dir: Path | None = None,
        node_bin: str = "node",
        decide_timeout: float = DEFAULT_DECIDE_TIMEOUT,
    ) -> None:
        resolved = (
            Path(engine_dir)
            if engine_dir is not None
            else Path(os.environ.get("BACA_ENGINE_DIR", "../slay-the-ceper"))
        ).resolve()
        bot_path = resolved / "src" / "logic" / "bots" / "HeuristicBot.js"
        if not bot_path.is_file():
            raise FileNotFoundError(f"HeuristicBot not found at {bot_path}")
        if not _RUNNER_SCRIPT.is_file():
            raise FileNotFoundError(f"Runner script missing at {_RUNNER_SCRIPT}")

        self._engine_dir = resolved
        self._decide_timeout = decide_timeout
        self._process = subprocess.Popen(  # noqa: S603 — node + our script under our control
            [node_bin, str(_RUNNER_SCRIPT), str(resolved)],
            cwd=str(resolved),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            text=False,
        )

        self._stdout_queue: queue.Queue[bytes] = queue.Queue()
        self._stderr_buffer: list[bytes] = []
        self._stderr_lock = threading.Lock()
        self._closed = threading.Event()

        self._stdout_reader = threading.Thread(
            target=self._pump_stdout, name="bc-bridge-stdout", daemon=True
        )
        self._stdout_reader.start()
        self._stderr_reader = threading.Thread(
            target=self._pump_stderr, name="bc-bridge-stderr", daemon=True
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

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._closed.is_set():
            raise RuntimeError("HeuristicBridge is closed")
        self._raise_if_stderr()
        self._raise_if_dead()

        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("HeuristicBridge stdin is not available")

        payload = json.dumps({"observation": observation}, separators=(",", ":"))
        try:
            stdin.write(payload.encode("utf-8") + b"\n")
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"HeuristicBridge stdin closed: {exc}") from exc

        try:
            line = self._stdout_queue.get(timeout=self._decide_timeout)
        except queue.Empty:
            self._raise_if_stderr()
            self._raise_if_dead()
            raise RuntimeError(
                f"HeuristicBridge.decide timed out after {self._decide_timeout:.1f}s"
            ) from None

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HeuristicBridge: malformed response {line!r}") from exc

        if "error" in message:
            raise RuntimeError(f"HeuristicBridge runner error: {message['error']}")
        action = message.get("action")
        if not isinstance(action, dict):
            raise RuntimeError(f"HeuristicBridge: missing action in {message!r}")
        return action

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._process.stdin is not None:
            with contextlib.suppress(BrokenPipeError, OSError):
                self._process.stdin.close()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2.0)

    def _pump_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        while not self._closed.is_set():
            line = stdout.readline()
            if not line:
                return
            self._stdout_queue.put(line)

    def _pump_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        while not self._closed.is_set():
            line = stderr.readline()
            if not line:
                return
            with self._stderr_lock:
                self._stderr_buffer.append(line)
            logger.debug("bc-bridge-stderr: %s", line.decode("utf-8", "replace").rstrip())

    def _raise_if_stderr(self) -> None:
        with self._stderr_lock:
            if self._stderr_buffer:
                combined = b"".join(self._stderr_buffer).decode("utf-8", "replace").strip()
                raise RuntimeError(f"HeuristicBridge runner wrote to stderr: {combined}")

    def _raise_if_dead(self) -> None:
        rc = self._process.poll()
        if rc is not None:
            raise RuntimeError(f"HeuristicBridge subprocess exited (returncode={rc})")
