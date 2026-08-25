"""Run each measurement phase in a fresh process.

Measuring the bandwidth sweep and the GEMM sweep in one interpreter gives the
wrong answer for whichever runs second, and the error is large: during
development the compute roof measured ~850 GFLOP/s in a fresh process but ~310
when it ran after the bandwidth sweep, a 2.7x understatement that dragged the
ridge point down with it.

The cause is not a leak in the measurement code. CPython's allocator holds on
to freed arenas, and numpy's own cache retains large blocks, so the several
hundred megabytes the bandwidth sweep touches are not returned to the OS when
the arrays are dropped. The next sweep then allocates under pressure on a
machine with a few gigabytes free and degrades into paging. ``gc.collect()``
does not fix it because the memory is free from Python's point of view -- it is
the OS that still considers it resident.

Process isolation is the reliable fix: a spawned interpreter starts with a
clean address space, and the OS reclaims everything when it exits. Each phase
therefore measures the machine rather than the residue of the previous phase.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any


def _entry(fn: Callable[..., Any], args: tuple, queue: mp.Queue) -> None:
    try:
        queue.put(("ok", fn(*args)))
    except Exception as exc:  # pragma: no cover - surfaced in the parent
        queue.put(("error", repr(exc)))


def run_isolated(fn: Callable[..., Any], *args: Any, timeout: float = 900.0) -> Any:
    """Call ``fn(*args)`` in a fresh spawned process and return its result.

    ``fn`` must be importable by name (a module-level function, not a lambda or
    closure) because the spawn start method pickles it. The return value must
    likewise be picklable.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_entry, args=(fn, args, queue))
    proc.start()
    try:
        status, payload = queue.get(timeout=timeout)
    finally:
        proc.join(timeout=10)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.terminate()
            proc.join()
    if status == "error":
        raise RuntimeError(f"isolated measurement failed: {payload}")
    return payload
