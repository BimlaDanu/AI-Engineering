"""Starting the front-of-graph model calls together, and being able not to.

Two claims are worth holding here and they pull in opposite directions. The first is
that overlapping the calls actually overlaps them, which needs a clock. The second is
that overlapping them changes nothing a reader sees -- same values, same exceptions,
same fallback when nothing was started -- which is what makes the switch safe to expose.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from src.agent.prefetch import MAX_IN_FLIGHT, Prefetch

# --------------------------------------------------------------------------
# Collecting a value, started or not
# --------------------------------------------------------------------------


def test_a_started_call_is_collected_rather_than_made_again() -> None:
    calls: list[str] = []

    def work() -> str:
        calls.append("ran")
        return "value"

    ahead = Prefetch()
    ahead.start("a", work)
    assert ahead.take("a", work) == "value"
    ahead.close()
    # The fallback passed to `take` is the same function. If it were invoked as well,
    # every prefetched call would be paid for twice -- which costs money rather than
    # merely time, and would not show up as a wrong answer anywhere.
    assert calls == ["ran"]


def test_a_call_nobody_started_is_simply_made_here() -> None:
    ahead = Prefetch()
    assert ahead.take("never-started", lambda: 7) == 7
    ahead.close()


def test_collecting_the_same_name_twice_makes_the_call_the_second_time() -> None:
    # There is no cache behind this, and there should not be: the second collector is a
    # node running a second time, and a node running twice is entitled to a fresh
    # answer rather than to a stale one nobody can see is stale.
    ahead = Prefetch()
    ahead.start("a", lambda: 1)
    assert ahead.take("a", lambda: 2) == 1
    assert ahead.take("a", lambda: 2) == 2
    ahead.close()


def test_starting_the_same_name_twice_starts_one_call() -> None:
    counter = itertools_count()
    ahead = Prefetch()
    ahead.start("a", counter)
    ahead.start("a", counter)
    assert ahead.take("a", counter) == 0
    ahead.close()


def itertools_count() -> Callable[[], int]:
    """A callable returning 0, 1, 2 ... so a repeated call is visible in the value.

    Written as a closure rather than reached for from the standard library because the
    thing under test is *how many times it was called*, and a value that counts is the
    only witness to that which survives being run on another thread.

    Returns:
        The callable.
    """
    seen = [0]
    lock = threading.Lock()

    def next_one() -> int:
        with lock:
            value = seen[0]
            seen[0] += 1
            return value

    return next_one


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------


def test_switched_off_nothing_is_started_and_the_value_is_still_right() -> None:
    calls: list[str] = []

    def work() -> str:
        calls.append("ran")
        return "value"

    ahead = Prefetch(enabled=False)
    ahead.start("a", work)
    assert calls == []
    assert ahead.take("a", work) == "value"
    assert calls == ["ran"]
    ahead.close()


def test_switched_off_no_thread_is_ever_created() -> None:
    # The point of the switch is to remove this module from the run, not to configure
    # it. A pool built and immediately idle would still be two threads and a shutdown
    # to get wrong, in a mode whose whole purpose is not to have them.
    before = threading.active_count()
    ahead = Prefetch(enabled=False)
    for name in ("a", "b", "c"):
        ahead.start(name, lambda: None)
    assert threading.active_count() == before
    ahead.close()


# --------------------------------------------------------------------------
# That it is actually concurrent
# --------------------------------------------------------------------------


def test_three_waiting_calls_take_the_time_of_one_rather_than_three() -> None:
    delay = 0.25

    def slow() -> float:
        time.sleep(delay)
        return delay

    ahead = Prefetch()
    started = time.perf_counter()
    for name in ("a", "b", "c"):
        ahead.start(name, slow)
    for name in ("a", "b", "c"):
        ahead.take(name, slow)
    elapsed = time.perf_counter() - started
    ahead.close()
    # Generous by design. The claim is "one round trip, not three", and asserting
    # anything tighter than half the sequential time would make this a test of how
    # loaded the machine running it happens to be.
    assert elapsed < 3 * delay * 0.6


def test_the_pool_is_wide_enough_for_the_calls_the_graph_makes() -> None:
    # A pool narrower than the work reintroduces the sequencing silently: every value
    # is still correct and the wait is still the sum, which is the one failure this
    # module exists to prevent and the one no assertion about values can catch.
    from src.agent.graph import CLARIFY_CALL, INTENT_CALL, PROPOSAL_CALL

    assert len({CLARIFY_CALL, INTENT_CALL, PROPOSAL_CALL}) <= MAX_IN_FLIGHT


# --------------------------------------------------------------------------
# Failure, which must arrive where it would have arrived anyway
# --------------------------------------------------------------------------


def test_an_exception_inside_a_started_call_surfaces_when_it_is_collected() -> None:
    def explode() -> None:
        raise RuntimeError("the provider said no")

    ahead = Prefetch()
    ahead.start("a", explode)
    with pytest.raises(RuntimeError, match="the provider said no"):
        ahead.take("a", explode)
    ahead.close()


def test_closing_with_calls_still_outstanding_does_not_raise() -> None:
    # The campaign that raised partway through. Nothing collects what was started, and
    # the shutdown in `run_campaign`'s `finally` has to cope with that rather than
    # leaving a non-daemon thread holding the process open.
    ahead = Prefetch()
    ahead.start("a", lambda: time.sleep(0.05))
    ahead.close()
    assert ahead.take("a", lambda: "fresh") == "fresh"
