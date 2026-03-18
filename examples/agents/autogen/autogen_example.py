"""
examples/agents/autogen/autogen_example.py
==========================================

Counter test mirroring the Academy example:
  - AssistantAgent increments a counter 5 times (values 1-5)
  - CriticAgent verifies the running total after each step
  - After the team run, openai_chat() interprets the final value (15 = 1+2+3+4+5)

Run
---
    OPENAI_API_KEY=sk-... python examples/agents/autogen/autogen_example.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import AsyncGenerator, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flowcept.agents.autogen.autogen_plugin import FlowceptAutoGenPlugin, openai_chat

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core.models import (
        ChatCompletionClient, CreateResult, RequestUsage,
        LLMMessage, AssistantMessage,
    )
    from autogen_core import CancellationToken
except ImportError:
    print("ERROR: pip install autogen-agentchat", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stub model client — canned counter responses, no API call needed
# ---------------------------------------------------------------------------

_STEPS = [1, 2, 3, 4, 5]
_RUNNING = 0
_STUB_RESPONSES = []
for _s in _STEPS:
    _RUNNING += _s
    _STUB_RESPONSES.append(
        f"assistant: increment({_s}) → running total = {_RUNNING}"
    )
_STUB_RESPONSES.append(
    f"critic: all 5 increments verified. Final value = {_RUNNING}. TERMINATE"
)

_stub_idx = 0


class _StubModelClient(ChatCompletionClient):
    """Returns canned counter responses without a real API call."""

    @property
    def model_info(self):
        from autogen_core.models import ModelInfo
        return ModelInfo(vision=False, function_calling=False, json_output=False,
                         family="stub", structured_output=False)

    @property
    def capabilities(self):
        from autogen_core.models import ModelCapabilities
        return ModelCapabilities(vision=False, function_calling=False, json_output=False)

    async def create(self, messages: Sequence[LLMMessage], *, tools=(),
                     tool_choice="auto", json_output=None,
                     extra_create_args=None, cancellation_token=None) -> CreateResult:
        global _stub_idx
        content = _STUB_RESPONSES[_stub_idx % len(_STUB_RESPONSES)]
        _stub_idx += 1
        return CreateResult(
            content=content,
            usage=RequestUsage(prompt_tokens=20, completion_tokens=20),
            finish_reason="stop",
            cached=False,
        )

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs) -> AsyncGenerator:
        result = await self.create(messages)
        yield result

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages, **kwargs) -> int:
        return 0

    def remaining_tokens(self, messages, **kwargs) -> int:
        return 4096

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Build the AutoGen team
# ---------------------------------------------------------------------------

def build_team(model_client) -> RoundRobinGroupChat:
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=(
            "You are a counter agent. Increment the counter by the given step "
            "and report the running total. When all 5 steps are done, end with TERMINATE."
        ),
    )
    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message=(
            "You are a verifier. Check each running total and confirm it is correct. "
            "When TERMINATE is reached, echo it."
        ),
    )
    termination = MaxMessageTermination(max_messages=len(_STUB_RESPONSES) + 1)
    return RoundRobinGroupChat(
        participants=[assistant, critic],
        termination_condition=termination,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(plugin: FlowceptAutoGenPlugin, team) -> int:
    task = "Increment a counter 5 times with step values 1, 2, 3, 4, 5 and report each running total."
    print(f"\n[example] Task: {task!r}\n", flush=True)

    result = await plugin.run_team(team, task, team_name="counter-team")

    # Parse final total from last assistant message
    final_value = 0
    for msg in result.messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            for token in content.split():
                try:
                    final_value = int(token)
                except ValueError:
                    pass

    print("\n" + "=" * 60)
    print("TEAM CONVERSATION")
    print("=" * 60)
    for msg in result.messages:
        source = getattr(msg, "source", "?")
        content = getattr(msg, "content", "")
        print(f"\n[{source}]\n{content}")
    print(f"\nStop reason: {result.stop_reason}")

    return final_value


def main() -> None:
    model_client = _StubModelClient()
    team = build_team(model_client)

    plugin = FlowceptAutoGenPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "autogen-counter-test",
            "performance_tracking": True,
        }
    )
    plugin.start()

    try:
        final_value = asyncio.run(_run(plugin, team))

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
            context={"agent": "post_team_analysis", "call_type": "interpret_result"},
        )
        print(f"\n[example] LLM says: {interpretation}\n", flush=True)
        print(f"[example] Counter reached {final_value}  (expected 15 = 1+2+3+4+5)", flush=True)

    finally:
        plugin.stop()


if __name__ == "__main__":
    main()
