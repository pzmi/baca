"""Unit tests for ``RpcClient`` failure-mode handling.

These tests bypass ``RpcClient.__init__`` (which spawns a real Node subprocess)
and construct a minimally-configured client with stub streams. They exercise
reader-thread death, call timeouts, subprocess death propagation, and stdin
write serialization.
"""

from __future__ import annotations

import io
import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from baca.rpc_client import (
    RpcClient,
    RpcTimeoutError,
    RpcTransportError,
)


class _FakeStdout:
    """Blocking ``read``-providing stub; unblocks when bytes are pushed or EOF set."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._cv = threading.Condition()
        self._eof = False

    def push(self, data: bytes) -> None:
        with self._cv:
            self._buf.extend(data)
            self._cv.notify_all()

    def set_eof(self) -> None:
        with self._cv:
            self._eof = True
            self._cv.notify_all()

    def read(self, _n: int) -> bytes:
        with self._cv:
            while not self._buf and not self._eof:
                self._cv.wait()
            if self._buf:
                out = bytes(self._buf)
                self._buf.clear()
                return out
            return b""


class _TrackingStdin:
    """Collects writes; flags interleaved concurrent writes as a failure."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_write = False
        self._concurrent_detected = False
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        with self._lock:
            if self._in_write:
                self._concurrent_detected = True
            self._in_write = True
        try:
            # Simulate a slow OS write to widen the race window.
            time.sleep(0.002)
            self.chunks.append(bytes(data))
            return len(data)
        finally:
            with self._lock:
                self._in_write = False

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def concurrent_detected(self) -> bool:
        return self._concurrent_detected


def _make_client(
    stdout: Any,
    stdin: Any | None = None,
    *,
    poll_returncode: int | None = None,
    default_timeout: float = 30.0,
) -> RpcClient:
    """Build an ``RpcClient`` without spawning a real subprocess."""
    client = RpcClient.__new__(RpcClient)
    proc = MagicMock()
    proc.stdin = stdin if stdin is not None else _TrackingStdin()
    proc.stdout = stdout
    proc.stderr = io.BytesIO(b"")
    proc.poll.return_value = poll_returncode
    proc.wait.return_value = poll_returncode if poll_returncode is not None else 0
    client._proc = proc
    client._default_timeout = default_timeout
    client._next_id = 1
    client._id_lock = threading.Lock()
    client._pending = {}
    client._pending_lock = threading.Lock()
    import queue as _queue

    client._notifications = _queue.Queue()
    client._closed = threading.Event()
    client._write_lock = threading.Lock()
    client._fatal_error = None
    client._fatal_lock = threading.Lock()
    return client


def _start_reader(client: RpcClient) -> threading.Thread:
    t = threading.Thread(target=client._reader_loop, name="rpc-reader-test", daemon=True)
    t.start()
    client._reader = t
    # stderr reader intentionally not started — _FakeStdout / BytesIO stderr is fine.
    client._stderr_reader = threading.Thread(target=lambda: None, daemon=True)
    client._stderr_reader.start()
    return t


def test_shouldRaiseWhenReaderThreadDiesBeforeResponse() -> None:
    # given — reader thread will raise on the first non-empty read
    class ExplodingStdout:
        def __init__(self) -> None:
            self._done = False

        def read(self, _n: int) -> bytes:
            if self._done:
                time.sleep(0.05)
                return b""
            self._done = True
            raise RuntimeError("boom from reader")

    stdout = ExplodingStdout()
    client = _make_client(stdout, default_timeout=2.0)
    reader = _start_reader(client)
    reader.join(timeout=1.0)
    assert not reader.is_alive()

    # when / then
    with pytest.raises(RpcTransportError) as exc_info:
        client.call("engine.ping", timeout=1.0)
    assert "boom from reader" in str(exc_info.value) or "failed state" in str(exc_info.value)


def test_shouldTimeoutWhenResponseNeverArrives() -> None:
    # given — stdout that never emits bytes and never closes
    stdout = _FakeStdout()
    client = _make_client(stdout, default_timeout=0.2)
    _start_reader(client)

    # when / then
    t0 = time.monotonic()
    with pytest.raises(RpcTimeoutError) as exc_info:
        client.call("engine.slow", timeout=0.2)
    elapsed = time.monotonic() - t0
    assert exc_info.value.method == "engine.slow"
    assert exc_info.value.timeout == pytest.approx(0.2, abs=0.1)
    assert elapsed < 1.5  # must not hang

    # cleanup reader
    stdout.set_eof()


def test_shouldRaiseWhenSubprocessExitsDuringPendingCall() -> None:
    # given — reader sees EOF (subprocess died), poll reports non-zero returncode
    stdout = _FakeStdout()
    stdin = _TrackingStdin()
    client = _make_client(stdout, stdin, poll_returncode=None, default_timeout=5.0)
    _start_reader(client)

    # when — a background caller is blocked on `.call`, then subprocess "dies"
    results: list[BaseException] = []

    def call_in_thread() -> None:
        try:
            client.call("engine.applyAction", timeout=5.0)
        except Exception as e:
            results.append(e)

    caller = threading.Thread(target=call_in_thread, daemon=True)
    caller.start()
    time.sleep(0.1)  # let the caller register and block

    client._proc.poll.return_value = 137  # type: ignore[attr-defined]
    stdout.set_eof()

    caller.join(timeout=2.0)

    # then
    assert not caller.is_alive(), "caller must unblock when subprocess dies"
    assert results, "caller must have raised"
    assert isinstance(results[0], RpcTransportError)
    assert "returncode" in str(results[0])


def test_shouldSerializeStdinWritesWhenCalledConcurrently() -> None:
    # given — stdin that flags interleaved writes; never emits responses
    stdout = _FakeStdout()
    stdin = _TrackingStdin()
    client = _make_client(stdout, stdin, default_timeout=0.1)
    _start_reader(client)

    errors: list[BaseException] = []

    def do_call(i: int) -> None:
        try:
            client.call(f"engine.method{i}", {"i": i}, timeout=0.1)
        except RpcTimeoutError:
            # expected — no response ever arrives; we only care about write ordering.
            pass
        except Exception as e:
            errors.append(e)

    # when — many threads issue calls concurrently
    threads = [threading.Thread(target=do_call, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    # then
    stdout.set_eof()
    assert not errors, f"unexpected errors: {errors}"
    assert not stdin.concurrent_detected, "stdin.write calls must be serialized"
    assert len(stdin.chunks) == 8, "each call must produce exactly one stdin chunk"
    # every chunk parses as a valid Content-Length framed JSON-RPC request
    for chunk in stdin.chunks:
        sep = chunk.find(b"\r\n\r\n")
        assert sep != -1
        body = chunk[sep + 4 :]
        parsed = json.loads(body)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["method"].startswith("engine.method")


def test_shouldFailPendingCallsWhenSubprocessStdoutClosesEarly() -> None:
    # given
    stdout = _FakeStdout()
    client = _make_client(stdout, poll_returncode=1, default_timeout=5.0)
    _start_reader(client)

    results: list[BaseException] = []

    def call_in_thread() -> None:
        try:
            client.call("engine.create", timeout=5.0)
        except Exception as e:
            results.append(e)

    caller = threading.Thread(target=call_in_thread, daemon=True)
    caller.start()
    time.sleep(0.1)

    # when — subprocess stdout closes unexpectedly
    stdout.set_eof()
    caller.join(timeout=2.0)

    # then
    assert not caller.is_alive()
    assert results
    assert isinstance(results[0], Exception)


def test_shouldRaiseImmediatelyOnNextCallAfterReaderDied() -> None:
    # given — reader that dies, then a subsequent call
    class ExplodingStdout:
        def read(self, _n: int) -> bytes:
            raise RuntimeError("unexpected read failure")

    client = _make_client(ExplodingStdout(), default_timeout=5.0)
    reader = _start_reader(client)
    reader.join(timeout=1.0)

    # when / then — second call should fail fast, not wait for timeout
    t0 = time.monotonic()
    with pytest.raises(RpcTransportError):
        client.call("engine.ping", timeout=5.0)
    assert time.monotonic() - t0 < 1.0
