"""Tests for process isolation of measurement phases.

These use module-level helpers rather than lambdas because the spawn start
method pickles the target by name.
"""

from __future__ import annotations

import os

import pytest

from prefill_decode_roofline.isolation import run_isolated


def _double(x: int) -> int:
    return x * 2


def _add(a: int, b: int) -> int:
    return a + b


def _get_pid() -> int:
    return os.getpid()


def _boom() -> None:
    raise ValueError("measurement exploded")


def test_returns_the_result_of_the_call():
    assert run_isolated(_double, 21) == 42


def test_passes_multiple_arguments():
    assert run_isolated(_add, 2, 3) == 5


def test_actually_runs_in_a_different_process():
    """The whole point: a fresh address space, not the caller's."""
    assert run_isolated(_get_pid) != os.getpid()


def test_exceptions_surface_in_the_parent():
    """A failed measurement must not look like a successful one."""
    with pytest.raises(RuntimeError, match="measurement exploded"):
        run_isolated(_boom)
