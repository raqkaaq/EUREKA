"""Run an owned parser/resolver child through bounded pipes, without runtime files."""

import os
import selectors
import subprocess
import time


class IsolatedProcessError(RuntimeError):
    def __init__(self, category: str):
        self.category = category
        super().__init__("Isolated worker exceeded a bound or could not complete.")


def run_bounded(argv: list[str], *, timeout_s: float, max_output_bytes: int) -> tuple[int, bytes]:
    deadline = time.monotonic() + timeout_s
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, close_fds=True)
    except OSError:
        raise IsolatedProcessError("failed") from None
    try:
        assert proc.stdout is not None
        output = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IsolatedProcessError("timeout")
                if not selector.select(remaining):
                    raise IsolatedProcessError("timeout")
                # Read at most one byte beyond the limit, never allocate an unbounded result.
                chunk = os.read(proc.stdout.fileno(), min(65536, max_output_bytes - len(output) + 1))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > max_output_bytes:
                    raise IsolatedProcessError("output_limit")
        try:
            code = proc.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise IsolatedProcessError("timeout") from None
        return code, bytes(output)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
