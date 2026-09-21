"""Start the front-of-graph model calls together instead of one after another.

Latency, and nothing else. Four nodes run before a single passage is retrieved --
``recall``, ``screen``, ``interpret``, ``formalise`` -- and three make a model
call: the restatement, the intent reading, and reading the chain's numbers out of
the sentence. Sequentially that is three round trips before the campaign has done
any work a reader would recognise as work.

They do not depend on each other. All three read the question as typed plus
whatever the conversation recalled, and none consumes another's output. The only
thing that must happen first is the screen's own gate, a set intersection costing
microseconds. So the wait becomes the slowest of the three rather than their sum.

No decision moves, no node is reordered, and no two nodes merge. Every node still
runs, still appears on the trace page, and still decides on the same evidence;
what changed is when the network call was issued. A node whose value was not
prefetched -- the flag is off, or it is being exercised alone in a test -- calls
the same function itself and cannot tell the difference, which is why the switch
is safe to expose.

Threads rather than ``asyncio``, because the calls are HTTP round trips and the
rest of the project is synchronous. Each call is submitted inside a copy of the
caller's :mod:`contextvars` context, which keeps it inside the campaign's trace
rather than surfacing as an unattached root run.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_IN_FLIGHT = 3
"""How many prefetched calls may be outstanding at once.

Exactly the number of model calls the front of the graph makes, rather than a round
number: a pool larger than the work is threads nobody uses, and a pool smaller than the
work quietly reintroduces the sequencing this module exists to remove.
"""


class Prefetch:
    """A place to start a call early and collect it later, or to do neither.

    Disabled is a first-class mode, not a degraded one. :meth:`start` then does
    nothing and :meth:`take` calls the function on the spot, which is the original
    sequential behaviour reached by the original code path -- so turning the switch off
    is a way of removing this module from the run rather than a way of configuring it.

    Not thread-safe against concurrent :meth:`start` calls for the same name, and it
    does not need to be: one node starts the calls and later nodes collect them.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Make a carrier for one campaign.

        Args:
            enabled: Whether to actually run calls ahead of time.
        """
        self.enabled = enabled
        self._executor: ThreadPoolExecutor | None = None
        self._started: dict[str, Future[object]] = {}

    def start(self, name: str, call: Callable[[], T]) -> None:
        """Begin a call now, to be collected under ``name`` later.

        Starting the same name twice is ignored rather than an error, so that a node
        which runs more than once in a campaign cannot issue a second copy of a call
        whose first copy nobody collected.

        Args:
            name: What :meth:`take` will ask for.
            call: The work, taking no arguments. Anything it needs is closed over,
                which keeps the decision about *what* to call at the call site where
                it is legible rather than in this module.
        """
        if not self.enabled or name in self._started:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=MAX_IN_FLIGHT, thread_name_prefix="prefetch"
            )
        context = contextvars.copy_context()
        self._started[name] = self._executor.submit(context.run, call)

    def take(self, name: str, call: Callable[[], T]) -> T:
        """Collect a started call, or make it now if none was started.

        An exception raised inside a prefetched call surfaces here, which is where it
        would have surfaced had the call been made in place. Swallowing it would turn a
        provider outage into a silently missing restatement, and a node that fell back
        to its deterministic path for a reason nobody could find.

        Args:
            name: The name it was started under.
            call: What to do if it was not started. Must be the same work.

        Returns:
            The value, from whichever route produced it.
        """
        pending = self._started.pop(name, None)
        if pending is None:
            return call()
        _logger.debug("prefetch_collected", extra={"call": name})
        return pending.result()  # type: ignore[return-value]

    def close(self) -> None:
        """Let go of the threads, waiting for anything still running.

        Called once at the end of a campaign. It waits rather than cancelling because a
        call already sent to a provider is already paid for, and because a thread pool
        abandoned mid-flight keeps a non-daemon thread alive and the process with it.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._started.clear()
