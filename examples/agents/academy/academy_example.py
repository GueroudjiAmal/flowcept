"""
examples/agents/academy/academy_example.py
==========================================

Academy example — two agents, FlowCept plugin, no LLM dependency.

Agents
------
CounterAgent  : holds an integer counter.  Actions: increment(n), get_value().
SummaryAgent  : increments the counter 5 times (1+2+3+4+5 = 15), then reads
                the final value.

Run
---
    # Thread executor (default):
    python examples/agents/academy/academy_example.py

    # Process executor — tests the ProcessPoolExecutor monkey-patch:
    ACADEMY_EXECUTOR=process python examples/agents/academy/academy_example.py

ProcessPoolExecutor monkey-patch test
--------------------------------------
When ACADEMY_EXECUTOR=process the example creates a plain ProcessPoolExecutor()
with NO explicit initializer.  The FlowceptAcademyPlugin monkey-patches
ProcessPoolExecutor.__init__ on start() so _worker_init is injected
automatically.  If the patch works, the performance report printed by stop()
will show action_emit / intercept_task / loop_emit / lifecycle_emit rows
(not just flush), confirming that provenance was captured inside the workers.
"""
from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


class CounterAgent(Agent):
    """Simple integer counter — two @action methods."""

    def __init__(self) -> None:
        super().__init__()
        self._value: int = 0

    @action
    async def increment(self, n: int = 1) -> int:
        """Add *n* to the counter and return the new value."""
        self._value += n
        return self._value

    @action
    async def get_value(self) -> int:
        """Return the current counter value."""
        return self._value


class SummaryAgent(Agent):
    """Increments the CounterAgent five times and records the final value."""

    def __init__(self) -> None:
        super().__init__()
        self._counter = None
        self._result: int = 0
        self._done: bool = False

    @action
    async def set_counter(self, counter) -> None:
        """Wire up the CounterAgent handle."""
        self._counter = counter

    @action
    async def get_result(self) -> int:
        return self._result

    @action
    async def is_done(self) -> bool:
        return self._done

    @loop
    async def summary_loop(self, shutdown: asyncio.Event) -> None:
        """Wait for the counter handle, increment 5 times, then finish."""
        while self._counter is None and not shutdown.is_set():
            await asyncio.sleep(0.05)
        if shutdown.is_set():
            return

        print("[SummaryAgent] Starting summary cycle …", flush=True)
        for step in range(1, 6):
            value = await self._counter.increment(step)
            print(f"[SummaryAgent]   increment({step}) → counter={value}", flush=True)

        self._result = await self._counter.get_value()
        print(f"[SummaryAgent] Final counter value: {self._result}", flush=True)
        self._done = True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _run(executor) -> None:
    exchange = LocalExchangeFactory()

    async with await Manager.from_exchange_factory(
        factory=exchange,
        executors=executor,
    ) as manager:
        counter = await manager.launch(CounterAgent)
        await counter.ping()

        summary = await manager.launch(SummaryAgent)
        await summary.ping()

        await summary.set_counter(counter)

        for _ in range(60):
            await asyncio.sleep(0.5)
            if await summary.is_done():
                break

        result = await summary.get_result()
        print(f"\n[driver] Counter reached {result}  (expected 15 = 1+2+3+4+5)", flush=True)
        assert result == 15, f"Expected 15 but got {result}"
        print("[driver] Assertion passed.", flush=True)


def main() -> None:
    init_logging("INFO")

    use_processes = os.environ.get("ACADEMY_EXECUTOR", "thread").lower() == "process"

    from flowcept import Flowcept

    with Flowcept():
        if use_processes:
            # Plain ProcessPoolExecutor — NO explicit initializer.
            # The monkey-patch installed by FlowceptAcademyPlugin.start() should
            # inject _worker_init automatically.  If provenance is captured
            # (action_emit / intercept_task rows appear in the perf report),
            # the patch is working.
            print("\n[main] Using ProcessPoolExecutor (monkey-patch test)", flush=True)
            executor = ProcessPoolExecutor(max_workers=4)
        else:
            print("\n[main] Using ThreadPoolExecutor", flush=True)
            executor = ThreadPoolExecutor(max_workers=4)

        asyncio.run(_run(executor))


if __name__ == "__main__":
    main()
