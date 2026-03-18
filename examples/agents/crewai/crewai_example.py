"""
examples/agents/crewai/crewai_example.py
=========================================

Counter test mirroring the Academy example:
  - CounterAgent increments a counter 5 times (values 1-5), reporting each step
  - SummaryAgent reads the final total and writes a short summary
  - After the crew, openai_chat() interprets the final value (15 = 1+2+3+4+5)

Run
---
    OPENAI_API_KEY=sk-... python examples/agents/crewai/crewai_example.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from flowcept.agents.crewai.crewai_plugin import FlowceptCrewAIPlugin, openai_chat

try:
    from crewai import Agent, Task, Crew, LLM
except ImportError:
    print("ERROR: pip install crewai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stub LLM — canned counter responses, no API key needed for the crew
# ---------------------------------------------------------------------------

def _make_fake_openai_completion(content: str):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types import CompletionUsage
    return ChatCompletion(
        id="chatcmpl-stub",
        choices=[Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(role="assistant", content=content),
            logprobs=None,
        )],
        created=1_700_000_000,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=30, prompt_tokens=60, total_tokens=90),
    )


_STUB_RESPONSES = [
    (
        "Final Answer: Counter incremented 5 times:\n"
        "  increment(1) → 1\n"
        "  increment(2) → 3\n"
        "  increment(3) → 6\n"
        "  increment(4) → 10\n"
        "  increment(5) → 15\n"
        "Final value: 15"
    ),
    (
        "Final Answer: The counter reached 15, which is the sum of the first "
        "5 positive integers (1+2+3+4+5 = 15), confirming correct increments."
    ),
]
_response_idx = 0


def _stub_create(*args, **kwargs):
    global _response_idx
    content = _STUB_RESPONSES[_response_idx % len(_STUB_RESPONSES)]
    _response_idx += 1
    return _make_fake_openai_completion(content)


# ---------------------------------------------------------------------------
# Build crew
# ---------------------------------------------------------------------------

def build_crew() -> Crew:
    llm = LLM(model="openai/gpt-4o-mini", api_key="stub-key")

    counter_agent = Agent(
        role="Counter",
        goal="Increment a counter 5 times with step values 1, 2, 3, 4, 5 and report each running total.",
        backstory=(
            "You are a precise counter agent. You add each step value to a running "
            "total and report the result after every increment."
        ),
        llm=llm,
        verbose=False,
        max_iter=2,
    )

    summary_agent = Agent(
        role="Summariser",
        goal="Read the counter result and write a one-sentence summary of what it represents.",
        backstory=(
            "You are a concise data analyst who turns numeric results into "
            "clear, plain-English explanations."
        ),
        llm=llm,
        verbose=False,
        max_iter=2,
    )

    count_task = Task(
        description=(
            "Increment a counter 5 times using step values 1, 2, 3, 4, 5. "
            "Report the running total after each increment and the final value."
        ),
        expected_output=(
            "A step-by-step report of each increment and the final counter value."
        ),
        agent=counter_agent,
    )

    summary_task = Task(
        description=(
            "Using the counter result from the previous task, write exactly one "
            "sentence explaining what the final value represents arithmetically."
        ),
        expected_output="One sentence explaining the arithmetic result.",
        agent=summary_agent,
        context=[count_task],
    )

    return Crew(
        agents=[counter_agent, summary_agent],
        tasks=[count_task, summary_task],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    plugin = FlowceptCrewAIPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "crewai-counter-test",
            "performance_tracking": True,
        }
    )
    plugin.start()

    crew = build_crew()

    print("\n[example] Running CrewAI crew …\n", flush=True)
    try:
        with patch(
            "openai.resources.chat.completions.Completions.create",
            side_effect=_stub_create,
        ):
            result = crew.kickoff()

        final_value = 15  # 1+2+3+4+5

        # Mirror academy's interpret_result @action
        print("\n[example] Calling openai_chat() to interpret the counter result …", flush=True)
        interpretation = openai_chat(
            prompt=(
                f"A counter was incremented 5 times with values 1, 2, 3, 4, 5 "
                f"and reached a final value of {final_value}. "
                f"In exactly one sentence, explain what this arithmetic result represents."
            ),
            model="gpt-4o-mini",
            system="You are a concise data analyst.",
            temperature=0.3,
            context={"agent": "post_crew_summary", "call_type": "interpret_result"},
        )
        print(f"\n[example] LLM says: {interpretation}\n", flush=True)
        print(f"[example] Counter reached {final_value}  (expected 15 = 1+2+3+4+5)", flush=True)

    finally:
        plugin.stop()

    print("\n" + "=" * 60)
    print("CREW OUTPUT")
    print("=" * 60)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
