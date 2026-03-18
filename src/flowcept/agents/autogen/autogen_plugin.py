# academy_coscientist/plugins/flowcept_autogen_plugin.py
"""
FlowCept provenance plugin for AutoGen (autogen_agentchat 0.7+) workflows.

Captures provenance for AutoGen team runs by consuming the ``run_stream()``
generator and recording every message as a FlowCept TaskObject, with the
overall team run as a WorkflowObject.

Provenance hierarchy produced:
  WorkflowObject  (one per team.run() call)
    └─ TaskObject  subtype=autogen_run      activity_id=<team_name>
         └─ TaskObject  subtype=autogen_message  activity_id=<agent_name>
              (one record per message yielded by run_stream)

Key FlowCept fields:
  task_id          — uuid per event
  workflow_id      — global shared workflow id (setdefault pattern)
  campaign_id      — from Flowcept.campaign_id
  parent_task_id   — messages are children of the enclosing run task
  group_id         — all tasks within one team.run() share a group_id
  activity_id      — team name | agent name | model name
  subtype          — autogen_run | autogen_message | autogen_result
  used / generated — message content, source, recipient
  status           — FINISHED | ERROR

Usage
-----
Standalone::

    plugin = FlowceptAutoGenPlugin(config={"workflow_name": "my-team"})
    plugin.start()
    result = asyncio.run(plugin.run_team(team, "Do something useful"))
    plugin.stop()

Or as a context manager::

    with FlowceptAutoGenPlugin(config={"workflow_name": "my-team"}) as plugin:
        result = asyncio.run(plugin.run_team(team, "task"))

Shared with Academy plugin::

    academy_plugin = FlowceptAcademyPlugin(config={...}).start()
    autogen_plugin = FlowceptAutoGenPlugin.from_academy_plugin(academy_plugin)
    result = asyncio.run(autogen_plugin.run_team(team, "task"))
    academy_plugin.stop()   # flushes the shared buffer
"""
from __future__ import annotations

import os

import asyncio
import time
import uuid
import threading
import logging
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance overhead timer
# ---------------------------------------------------------------------------

class _ProvenanceStats:
    __slots__ = ("_lock", "_counts", "_totals", "_mins", "_maxs", "_raw")

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._totals: dict[str, float] = {}
        self._mins:   dict[str, float] = {}
        self._maxs:   dict[str, float] = {}
        self._raw: list[tuple[str, str, float]] = []

    def record(self, category: str, elapsed: float) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            if category not in self._counts:
                self._counts[category] = 0
                self._totals[category] = 0.0
                self._mins[category]   = float("inf")
                self._maxs[category]   = 0.0
            self._counts[category] += 1
            self._totals[category] += elapsed
            if elapsed < self._mins[category]:
                self._mins[category] = elapsed
            if elapsed > self._maxs[category]:
                self._maxs[category] = elapsed
            self._raw.append((ts, category, elapsed))

    def summary(self) -> str:
        col = 22
        header = (
            f"{'Category':<{col}} {'N':>7} {'Total(ms)':>11} "
            f"{'Mean(µs)':>9} {'Min(µs)':>8} {'Max(µs)':>8}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        with self._lock:
            for cat in sorted(self._counts):
                n     = self._counts[cat]
                total = self._totals[cat]
                mean  = (total / n) if n else 0.0
                mn    = self._mins.get(cat, 0.0)
                mx    = self._maxs.get(cat, 0.0)
                rows.append(
                    f"{cat:<{col}} {n:>7} {total*1e3:>11.3f} "
                    f"{mean*1e6:>9.1f} {mn*1e6:>8.1f} {mx*1e6:>8.1f}"
                )
        return "\n".join(rows)

    def to_csv(self, path: str, workflow_id: str | None = None) -> None:
        import csv
        write_header = not os.path.exists(path)
        with self._lock:
            raw_snapshot = list(self._raw)
        wf = workflow_id or ""
        rows = [
            {
                "timestamp_utc": ts,
                "workflow_id":   wf,
                "category":      cat,
                "elapsed_us":    round(elapsed * 1e6, 3),
            }
            for ts, cat, elapsed in raw_snapshot
        ]
        fieldnames = ["timestamp_utc", "workflow_id", "category", "elapsed_us"]
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Standalone interceptor wrapper
# ---------------------------------------------------------------------------

class _AutoGenInterceptor:
    """Standalone FlowCept interceptor for use without an Academy plugin."""

    def __init__(self) -> None:
        self._interceptor = None
        self._workflow_id: str | None = None
        self._campaign_id: str | None = None

    def start(self, workflow_name: str, campaign_id: str | None = None) -> None:
        from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor
        from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject

        self._workflow_id = str(uuid.uuid4())
        self._campaign_id = campaign_id or str(uuid.uuid4())

        self._interceptor = BaseInterceptor(kind="autogen")
        self._interceptor.start(
            bundle_exec_id=self._workflow_id,
            check_safe_stops=False,
        )

        wf = WorkflowObject()
        wf.workflow_id = self._workflow_id
        wf.campaign_id = self._campaign_id
        wf.name = workflow_name
        self._interceptor.send_workflow_message(wf)

    def stop(self) -> None:
        if self._interceptor is None:
            return
        try:
            self._interceptor.stop(check_safe_stops=False)
        except Exception as e:
            _log.warning("Interceptor stop error: %r", e)
        self._interceptor = None

    def send_team_workflow(self, team_name: str, group_id: str) -> str:
        if self._interceptor is None:
            return str(uuid.uuid4())
        from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
        wf = WorkflowObject()
        wf.workflow_id = str(uuid.uuid4())
        wf.name = team_name
        wf.campaign_id = self._campaign_id
        wf.parent_workflow_id = self._workflow_id
        wf.custom_metadata = {"group_id": group_id, "framework": "autogen"}
        self._interceptor.send_workflow_message(wf)
        return wf.workflow_id

    # Accept the same name used by AcademyInterceptor so shared interceptors work
    send_graph_workflow = send_team_workflow

    def intercept_task(self, task_dict: dict) -> None:
        if self._interceptor is None:
            return
        from flowcept.commons.flowcept_dataclasses.task_object import TaskObject
        from flowcept.commons.vocabulary import Status

        task_dict.setdefault("task_id", str(uuid.uuid4()))
        task_dict.setdefault("workflow_id", self._workflow_id)
        task_dict.setdefault("campaign_id", self._campaign_id)

        raw = task_dict.get("status", "FINISHED")
        if isinstance(raw, str):
            try:
                task_dict["status"] = Status[raw].value
            except KeyError:
                task_dict["status"] = Status.FINISHED.value

        TaskObject.enrich_task_dict(task_dict)
        self._interceptor.intercept(task_dict)


# ---------------------------------------------------------------------------
# Provenance stream runner
# ---------------------------------------------------------------------------

def _safe_clip(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_clip(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_clip(v, depth + 1) for v in obj]
    # AutoGen message objects
    try:
        if hasattr(obj, "model_dump"):
            return _safe_clip(obj.model_dump(), depth + 1)
    except Exception:
        pass
    try:
        if hasattr(obj, "__dict__"):
            return _safe_clip(vars(obj), depth + 1)
    except Exception:
        pass
    return repr(obj)


async def _run_with_provenance(
    team: Any,
    task: str,
    interceptor: Any,
    stats: _ProvenanceStats | None,
    team_name: str = "autogen_team",
    source_agent_id: str | None = None,
) -> Any:
    """
    Consume team.run_stream() and emit a FlowCept provenance record for each
    message plus one overall run record.

    Returns the final TaskResult.
    """
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.messages import BaseChatMessage, BaseAgentEvent
    from autogen_core import CancellationToken

    group_id    = str(uuid.uuid4())
    run_task_id = str(uuid.uuid4())
    run_start   = time.time()

    # Emit a sub-WorkflowObject for this team run
    interceptor.send_graph_workflow(team_name, group_id)

    custom_meta: dict = {"team_name": team_name, "framework": "autogen"}
    if source_agent_id:
        custom_meta["source_agent_id"] = source_agent_id

    messages_captured: list[dict] = []
    final_result: Any = None

    t_stream_start = time.perf_counter()

    try:
        stream = team.run_stream(task=task, cancellation_token=CancellationToken())
        async for item in stream:
            t0 = time.perf_counter()
            if isinstance(item, TaskResult):
                final_result = item
            else:
                # Each item is a BaseChatMessage or BaseAgentEvent
                msg_task_id = str(uuid.uuid4())
                source      = getattr(item, "source", None) or "unknown"
                content     = getattr(item, "content", None)
                msg_type    = type(item).__name__

                # Serialize content
                if isinstance(content, str):
                    content_clip = content
                else:
                    content_clip = _safe_clip(content)

                msg_record: dict = {
                    "task_id":      msg_task_id,
                    "subtype":      "autogen_message",
                    "activity_id":  source,
                    "group_id":     group_id,
                    "parent_task_id": run_task_id,
                    "started_at":   time.time(),
                    "ended_at":     time.time(),
                    "status":       "FINISHED",
                    "used":         {"task": task, "agent": source},
                    "generated":    {"content": content_clip, "message_type": msg_type},
                    "custom_metadata": {
                        "agent_name":   source,
                        "message_type": msg_type,
                        "framework":    "autogen",
                    },
                }
                interceptor.intercept_task(msg_record)
                messages_captured.append({
                    "source":  source,
                    "content": content_clip[:200] if isinstance(content_clip, str) else content_clip,
                })
                if stats is not None:
                    stats.record("message_intercept", time.perf_counter() - t0)

    except Exception as exc:
        # Emit the run record as ERROR then re-raise
        run_task: dict = {
            "task_id":     run_task_id,
            "subtype":     "autogen_run",
            "activity_id": team_name,
            "group_id":    group_id,
            "started_at":  run_start,
            "ended_at":    time.time(),
            "status":      "ERROR",
            "used":        {"task": task},
            "generated":   {"messages": messages_captured},
            "stderr":      str(exc),
            "custom_metadata": custom_meta,
        }
        interceptor.intercept_task(run_task)
        raise

    if stats is not None:
        stats.record("stream_total", time.perf_counter() - t_stream_start)

    # Emit the overall run record (ONE complete record with inputs + outputs)
    summary = _safe_clip(getattr(final_result, "stop_reason", None) or "completed")
    msg_count = len(messages_captured)
    last_msg = messages_captured[-1]["content"] if messages_captured else ""

    run_task = {
        "task_id":     run_task_id,
        "subtype":     "autogen_run",
        "activity_id": team_name,
        "group_id":    group_id,
        "started_at":  run_start,
        "ended_at":    time.time(),
        "status":      "FINISHED",
        "used":        {"task": task},
        "generated":   {
            "stop_reason":   summary,
            "message_count": msg_count,
            "last_message":  last_msg,
            "messages":      messages_captured,
        },
        "custom_metadata": custom_meta,
    }
    interceptor.intercept_task(run_task)

    return final_result


# ---------------------------------------------------------------------------
# Module-level active interceptor — set by start() / from_academy_plugin()
# ---------------------------------------------------------------------------

_ACTIVE_INTERCEPTOR = None


def record_llm_call(payload: dict) -> None:
    """
    Public API to record an LLM call into the active FlowCept provenance graph.

    Converts the payload into a TaskObject (subtype=llm_call) and routes it
    through the active interceptor.  No-ops if the plugin has not been started.

    Minimum payload keys:
        type   : "chat_completion"
        model  : str
        text   : str
        usage  : dict with prompt_tokens / completion_tokens / total_tokens
    """
    interceptor = _ACTIVE_INTERCEPTOR
    if interceptor is None:
        return
    import uuid as _uuid
    import time as _time

    elapsed = payload.get("elapsed_s", 0.0)
    now = _time.time()
    model = payload.get("model_used") or payload.get("model", "unknown")

    used: dict = {}
    for k in ("model", "model_used", "messages", "user_prompt", "system_prompt",
              "temperature", "top_p", "max_tokens", "reasoning_effort",
              "tools_provided", "tool_choice", "stop_sequences", "top_k",
              "thinking_budget_tokens"):
        if k in payload:
            used[k] = payload[k]
    if payload.get("temperature_suppressed"):
        used["temperature_suppressed"] = True

    generated: dict = {}
    for k in ("text", "finish_reason", "stop_reason", "stop_sequence",
              "tool_calls", "tool_uses", "thinking_text",
              "system_fingerprint", "response_id", "usage", "elapsed_s"):
        if k in payload:
            generated[k] = payload[k]
    if "error" in payload:
        generated["error"] = str(payload["error"])

    task: dict = {
        "task_id":      str(_uuid.uuid4()),
        "subtype":      "llm_call",
        "activity_id":  model,
        "started_at":   now - elapsed,
        "ended_at":     now,
        "status":       "ERROR" if "error" in payload else "FINISHED",
        "used":         used,
        "generated":    generated,
        "custom_metadata": {
            "model":     model,
            "framework": payload.get("context", {}).get("framework", ""),
            "context":   payload.get("context", {}),
        },
    }
    interceptor.intercept_task(task)


def openai_chat(
    prompt: str,
    model: str = "gpt-4o-mini",
    system: str = "You are a helpful assistant.",
    temperature: float | None = 0.3,
    top_p: float | None = None,
    max_tokens: int | None = None,
    n: int = 1,
    stop: list[str] | str | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    user: str | None = None,
    context: dict | None = None,
) -> str:
    """
    Make an OpenAI chat completion call and record it for FlowCept provenance.

    Captures all request parameters and response fields as a child TaskObject.
    No-ops gracefully if OPENAI_API_KEY is not set or the plugin is not started.
    """
    import openai as _openai
    import os as _os
    import time as _time

    client = _openai.OpenAI(api_key=_os.environ.get("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    req: dict = {"model": model, "messages": messages, "n": n}
    if temperature is not None:
        req["temperature"] = temperature
    if top_p is not None:
        req["top_p"] = top_p
    if max_tokens is not None:
        req["max_completion_tokens"] = max_tokens
    if stop is not None:
        req["stop"] = stop
    if frequency_penalty != 0.0:
        req["frequency_penalty"] = frequency_penalty
    if presence_penalty != 0.0:
        req["presence_penalty"] = presence_penalty
    if seed is not None:
        req["seed"] = seed
    if reasoning_effort is not None:
        req["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        req["response_format"] = response_format
    if tools is not None:
        req["tools"] = tools
    if tool_choice is not None:
        req["tool_choice"] = tool_choice
    if user is not None:
        req["user"] = user

    t0 = _time.time()
    response = client.chat.completions.create(**req)
    elapsed = _time.time() - t0

    choice = response.choices[0]
    text = choice.message.content or ""
    usage = response.usage or {}

    usage_dict: dict = {}
    for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, attr):
            usage_dict[attr] = getattr(usage, attr)
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        ctd = usage.completion_tokens_details
        usage_dict["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", None)
        usage_dict["accepted_prediction_tokens"] = getattr(ctd, "accepted_prediction_tokens", None)
        usage_dict["rejected_prediction_tokens"] = getattr(ctd, "rejected_prediction_tokens", None)
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        ptd = usage.prompt_tokens_details
        usage_dict["cached_tokens"] = getattr(ptd, "cached_tokens", None)

    tool_calls = None
    if choice.message.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in choice.message.tool_calls
        ]

    record_llm_call({
        "type": "chat_completion",
        "model": model,
        "model_used": response.model,
        "messages": messages,
        "user_prompt": prompt,
        "system_prompt": system,
        "temperature": temperature,
        "temperature_suppressed": temperature is None,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "stop": stop,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "tools_provided": [t.get("function", {}).get("name") for t in (tools or [])],
        "tool_choice": tool_choice,
        "text": text,
        "finish_reason": choice.finish_reason,
        "tool_calls": tool_calls,
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "response_id": response.id,
        "created": response.created,
        "usage": usage_dict,
        "elapsed_s": elapsed,
        "context": context or {},
    })
    return text


def anthropic_chat(
    prompt: str,
    model: str = "claude-3-5-haiku-latest",
    system: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float | None = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    stop_sequences: list[str] | None = None,
    tools: list | None = None,
    tool_choice: dict | None = None,
    thinking: dict | None = None,
    metadata: dict | None = None,
    context: dict | None = None,
) -> str:
    """
    Make an Anthropic (Claude) chat completion call and record it for FlowCept provenance.

    Captures all request/response fields — including thinking blocks, tool use,
    and cache token counts — as a child TaskObject.
    """
    import anthropic as _anthropic
    import os as _os
    import time as _time

    client = _anthropic.Anthropic(api_key=_os.environ.get("ANTHROPIC_API_KEY"))
    req: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        req["temperature"] = temperature
    if top_p is not None:
        req["top_p"] = top_p
    if top_k is not None:
        req["top_k"] = top_k
    if stop_sequences:
        req["stop_sequences"] = stop_sequences
    if tools:
        req["tools"] = tools
    if tool_choice:
        req["tool_choice"] = tool_choice
    if thinking:
        req["thinking"] = thinking
    if metadata:
        req["metadata"] = metadata

    t0 = _time.time()
    response = client.messages.create(**req)
    elapsed = _time.time() - t0

    text = ""
    thinking_text = ""
    tool_uses = []
    for block in response.content:
        if block.type == "text":
            text += block.text
        elif block.type == "thinking":
            thinking_text += getattr(block, "thinking", "")
        elif block.type == "tool_use":
            tool_uses.append({"id": block.id, "name": block.name, "input": block.input})

    usage = response.usage
    usage_dict = {
        "input_tokens":                  getattr(usage, "input_tokens", None),
        "output_tokens":                 getattr(usage, "output_tokens", None),
        "cache_creation_input_tokens":   getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens":       getattr(usage, "cache_read_input_tokens", None),
    }

    record_llm_call({
        "type": "chat_completion",
        "model": model,
        "model_used": response.model,
        "messages": req["messages"],
        "user_prompt": prompt,
        "system_prompt": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "temperature_suppressed": temperature is None,
        "top_p": top_p,
        "top_k": top_k,
        "stop_sequences": stop_sequences,
        "thinking_budget_tokens": (thinking or {}).get("budget_tokens"),
        "tools_provided": [t.get("name") for t in (tools or [])],
        "tool_choice": tool_choice,
        "text": text,
        "thinking_text": thinking_text if thinking_text else None,
        "finish_reason": response.stop_reason,
        "stop_sequence": response.stop_sequence,
        "tool_uses": tool_uses if tool_uses else None,
        "response_id": response.id,
        "usage": usage_dict,
        "elapsed_s": elapsed,
        "context": context or {},
    })
    return text


# ---------------------------------------------------------------------------
# Public plugin class
# ---------------------------------------------------------------------------

class FlowceptAutoGenPlugin:
    """
    FlowCept provenance plugin for AutoGen (autogen_agentchat 0.7+) team runs.

    Wraps ``team.run_stream()`` to capture provenance without patching the
    AutoGen library.  Each message yielded by the stream becomes a TaskObject
    child of the overall run TaskObject.

    Parameters
    ----------
    config : dict, optional
        Plugin configuration keys:
          enabled              (bool, default True)
          workflow_name        (str, default "autogen-workflow")
          performance_tracking (bool, default True)
          perf_csv             (str, optional) — explicit path for timing CSV.

    Usage
    -----
    Standalone::

        plugin = FlowceptAutoGenPlugin(config={"workflow_name": "my-team"})
        plugin.start()
        result = asyncio.run(plugin.run_team(team, "Solve the problem"))
        plugin.stop()

    Shared with Academy plugin::

        academy_plugin = FlowceptAcademyPlugin(config={...}).start()
        autogen_plugin = FlowceptAutoGenPlugin.from_academy_plugin(academy_plugin)
        result = asyncio.run(autogen_plugin.run_team(team, "task"))
        academy_plugin.stop()
    """

    def __init__(self, config: dict | None = None, _shared_interceptor=None) -> None:
        cfg = config or {}
        self._enabled:        bool       = cfg.get("enabled", True)
        self._workflow_name:  str        = cfg.get("workflow_name", "autogen-workflow")
        self._campaign_id:    str | None = cfg.get("campaign_id", None)
        self._perf_tracking:  bool       = cfg.get("performance_tracking", True)
        self._perf_csv:       str | None  = cfg.get("perf_csv", None)
        self._shared_interceptor = _shared_interceptor
        self._interceptor    = _shared_interceptor or _AutoGenInterceptor()
        self._owns_interceptor: bool = _shared_interceptor is None
        self._stats: _ProvenanceStats | None = None
        self._started = False

    @classmethod
    def from_academy_plugin(
        cls,
        academy_plugin: Any,
        config: dict | None = None,
    ) -> "FlowceptAutoGenPlugin":
        """
        Create an AutoGen plugin that shares the buffer of a running
        FlowceptAcademyPlugin.
        """
        interceptor = academy_plugin._interceptor
        inst = cls(config=config, _shared_interceptor=interceptor)
        inst._started = True
        global _ACTIVE_INTERCEPTOR
        _ACTIVE_INTERCEPTOR = interceptor

        inst._stats = _ProvenanceStats() if (config or {}).get("performance_tracking", True) else None
        return inst

    def start(self) -> "FlowceptAutoGenPlugin":
        if not self._enabled or self._started:
            return self
        if not self._owns_interceptor:
            return self
        try:
            self._stats = _ProvenanceStats() if self._perf_tracking else None
            self._interceptor.start(self._workflow_name, campaign_id=self._campaign_id)
            self._started = True
            global _ACTIVE_INTERCEPTOR
            _ACTIVE_INTERCEPTOR = self._interceptor
            wf_id = self._interceptor._workflow_id
            print(
                f"[FlowceptAutoGenPlugin] Started\n"
                f"  workflow_id : {wf_id}\n"
                f"  campaign_id : {self._interceptor._campaign_id}\n"
                f"  Capturing   : team run (sub-workflow), messages (child tasks).",
                flush=True,
            )
        except Exception as e:
            print(
                f"[FlowceptAutoGenPlugin] WARNING: failed to start — {e!r}. "
                "Continuing without provenance capture.",
                flush=True,
            )
            _log.exception("FlowceptAutoGenPlugin start failed")
            self._enabled = False
        return self

    async def run_team(
        self,
        team: Any,
        task: str,
        team_name: str | None = None,
        source_agent_id: str | None = None,
    ) -> Any:
        """
        Run a team and capture provenance for the entire conversation.

        Parameters
        ----------
        team : RoundRobinGroupChat | SelectorGroupChat | any team with run_stream
            The AutoGen team to run.
        task : str
            The task / initial message to send to the team.
        team_name : str, optional
            Human-readable name for this run in provenance records.
            Defaults to the team's ``name`` attribute or "autogen_team".
        source_agent_id : str, optional
            ID of an upstream agent (e.g. Academy AgentId) that produced the
            input data.  Stored in custom_metadata for cross-framework linkage.

        Returns
        -------
        TaskResult
            The final result returned by AutoGen.
        """
        name = team_name or getattr(team, "name", None) or "autogen_team"
        if not self._started:
            # plugin not started — run the team without provenance
            from autogen_agentchat.base import TaskResult
            from autogen_core import CancellationToken
            results = []
            async for item in team.run_stream(task=task, cancellation_token=CancellationToken()):
                if isinstance(item, TaskResult):
                    results.append(item)
            return results[-1] if results else None

        return await _run_with_provenance(
            team=team,
            task=task,
            interceptor=self._interceptor,
            stats=self._stats,
            team_name=name,
            source_agent_id=source_agent_id,
        )

    def stop(self) -> None:
        if not self._started:
            return
        if not self._owns_interceptor:
            self._started = False
            print(
                "[FlowceptAutoGenPlugin] Detached from shared buffer "
                "(flushed by the owning plugin).",
                flush=True,
            )
            self._maybe_write_perf_csv()
            return
        try:
            self._interceptor.stop()
        except Exception as e:
            print(f"[FlowceptAutoGenPlugin] Warning during stop: {e!r}", flush=True)
        self._started = False
        print("[FlowceptAutoGenPlugin] Stopped.", flush=True)
        self._maybe_write_perf_csv()
        global _ACTIVE_INTERCEPTOR
        _ACTIVE_INTERCEPTOR = None

    def _maybe_write_perf_csv(self) -> None:
        if self._stats is None:
            return
        print(
            "\n[FlowceptAutoGenPlugin] Provenance overhead report:\n"
            + self._stats.summary()
            + "\n  (N = event count; Total/Mean/Min/Max in ms/µs respectively)\n",
            flush=True,
        )
        wf_id = self._interceptor._workflow_id
        csv_path = self._perf_csv or f"autogen_provenance_perf_{wf_id}.csv"
        try:
            self._stats.to_csv(csv_path, workflow_id=wf_id)
            print(f"[FlowceptAutoGenPlugin] Performance stats → {csv_path}", flush=True)
        except Exception as e:
            print(f"[FlowceptAutoGenPlugin] Warning: could not write perf CSV — {e!r}", flush=True)

    def __enter__(self) -> "FlowceptAutoGenPlugin":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()
