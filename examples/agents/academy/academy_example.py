"""
examples/agents/academy/academy_example.py
==========================================

Academy example — two agents, FlowCept plugin, LLM call.

Agents
------
CounterAgent   : holds an integer counter.  Actions: increment(n), get_value().
SummaryAgent   : increments the counter 5 times (1+2+3+4+5 = 15), then calls
               openai_chat() to interpret the result.

Run
---
    OPENAI_API_KEY=sk-... python examples/agents/academy/academy_example.py

Plugin configuration
--------------------
The FlowceptAcademyPlugin can be activated without any code changes by enabling
it in your settings.yaml:

    plugins:
      academy:
        enabled: true
        kind: academy
        workflow_name: "academy-counter-example"
        performance_tracking: true

When enabled this way, Flowcept.start() / .stop() (or the context manager)
will automatically start and stop the plugin — no explicit plugin.start() /
plugin.stop() calls needed in your code.
"""
from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from flowcept.agents.academy.academy_plugin import openai_chat


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
    """
    Calls CounterAgent.increment() several times, reads the total,
    then calls an LLM to interpret the result.
    """

    def __init__(self) -> None:
        super().__init__()
        self._counter = None
        self._result: int = 0
        self._llm_summary: str = ""
        self._done: bool = False

    @action
    async def set_counter(self, counter) -> None:
        """Wire up the CounterAgent handle."""
        self._counter = counter

    @action
    async def get_result(self) -> int:
        return self._result

    @action
    async def get_llm_summary(self) -> str:
        return self._llm_summary

    @action
    async def is_done(self) -> bool:
        return self._done

    @action
    async def interpret_result(self, value: int) -> str:
        """
        Ask the LLM to interpret the counter value.

        openai_chat() makes the API call and automatically fires record_llm_call()
        so the plugin captures this as a child TaskObject (subtype=llm_call)
        linked to this @action via parent_task_id.
        """
        prompt = (
            f"The counter was incremented 5 times with values 1, 2, 3, 4, 5 "
            f"and reached a final value of {value}. "
            f"In exactly one sentence, explain what this arithmetic result represents."
        )
        summary = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai_chat(
                prompt=prompt,
                model="gpt-4o-mini",
                system="You are a concise data analyst.",
                temperature=0.3,
                context={"agent": "SummaryAgent", "call_type": "interpret_result"},
            ),
        )
        print(f"[SummaryAgent] LLM says: {summary}", flush=True)
        return summary

    @loop
    async def summary_loop(self, shutdown: asyncio.Event) -> None:
        """Wait for the counter handle, increment 5 times, then call the LLM."""
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

        self._llm_summary = await self.interpret_result(self._result)
        self._done = True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _run() -> None:
    exchange = LocalExchangeFactory()
    executor = ThreadPoolExecutor(max_workers=4)

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
        llm_summary = await summary.get_llm_summary()
        print(f"\n[driver] Counter reached {result}  (expected 15 = 1+2+3+4+5)", flush=True)
        print(f"[driver] LLM summary: {llm_summary}\n", flush=True)
        assert result == 15, f"Expected 15 but got {result}"
        print("[driver] Assertion passed.", flush=True)


def main() -> None:
    init_logging("INFO")

    # The academy plugin is configured in settings.yaml under plugins.academy.
    # Flowcept reads that config and auto-starts/stops the plugin — no explicit
    # plugin.start() / plugin.stop() calls needed here.
    from flowcept import Flowcept

    with Flowcept():
        asyncio.run(_run())


if __name__ == "__main__":
    main()
