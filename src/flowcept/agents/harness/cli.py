"""The ``flowcept-harness`` command.

Everything outside the capture path lives here: inspecting what was captured,
pushing it into Flowcept, and wiring harnesses up in the first place. Unlike
the hook path this is not latency-sensitive, so it imports freely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flowcept.version import __version__

from .config import Config, load_config

# Exit codes. Anything non-zero means the *command* failed; capture failures
# are always silent by design.
OK = 0
FAILED = 1


# -- helpers -----------------------------------------------------------------


def _buffers(config: Config) -> list[Path]:
    if not config.buffers_dir.is_dir():
        return []
    return sorted(config.buffers_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _resolve_inputs(config: Config, given: list[str] | None, want_all: bool) -> list[Path]:
    if given:
        return [Path(p).expanduser() for p in given]
    if want_all:
        return _buffers(config)
    latest = _buffers(config)
    return latest[:1]


def _summarize(path: Path) -> dict[str, Any]:
    counts = {"workflow": 0, "task": 0, "agent": 0, "other": 0}
    session: dict[str, Any] = {}
    for record in _read_records(path):
        kind = record.get("type")
        counts[kind if kind in counts else "other"] += 1
        if kind == "workflow" and record.get("parent_workflow_id") is None:
            session = record
    return {
        "path": path,
        "counts": counts,
        "workflow_id": session.get("workflow_id") or path.stem,
        "status": session.get("status") or "UNKNOWN",
        "name": session.get("name") or "?",
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "generated": session.get("generated") or {},
    }


def _fmt_time(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    import datetime

    return datetime.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")


# -- commands ----------------------------------------------------------------


def cmd_hook(args, config: Config) -> int:
    """Route a hook payload from stdin to the right adapter.

    Lets one binary serve every harness: ``flowcept-harness hook --harness
    codex`` is a valid hook command anywhere a program can be named.
    """
    from .runtime import read_stdin_json, run_capture

    def _run(cfg: Config) -> None:
        payload = read_stdin_json()
        if args.harness == "claude_code":
            from flowcept.agents.claude_code import claude_code_plugin as claude_code

            if args.event:
                payload["hook_event_name"] = payload.get("hook_event_name") or args.event
            claude_code.handle(payload, cfg)
        else:
            from flowcept.agents.cli_harness import cli_harness_plugin as generic

            generic.handle(payload, cfg, harness=args.harness, profile=args.profile, event=args.event)

    return run_capture(_run, config)


def cmd_status(args, config: Config) -> int:
    buffers = _buffers(config)
    print(f"flowcept-harness {__version__}")
    print(f"  enabled:      {config.enabled}")
    print(f"  home:         {config.home}")
    print(f"  buffers:      {config.buffers_dir} ({len(buffers)} session(s))")
    print(f"  content mode: {config.content_mode}   redact: {config.redact}")
    print(f"  online:       {config.online}")

    try:
        import flowcept

        print(f"  flowcept:     {getattr(flowcept, '__version__', 'installed')}")
    except ImportError:
        print("  flowcept:     not installed (capture works; ingest and reports do not)")
        return OK

    if args.check_backend:
        try:
            from flowcept.commons.daos.mq_dao.mq_dao_base import MQDao

            MQDao.build()
            print("  backend:      reachable")
        except Exception as exc:
            print(f"  backend:      unreachable ({type(exc).__name__}: {exc})")
    return OK


def cmd_sessions(args, config: Config) -> int:
    buffers = _buffers(config)
    if not buffers:
        print(f"No sessions captured yet under {config.buffers_dir}")
        return OK

    for path in buffers[: args.limit]:
        info = _summarize(path)
        gen = info["generated"]
        detail = " ".join(f"{k}={v}" for k, v in gen.items()) or "-"
        print(f"{info['workflow_id'][:8]}  {info['status']:8}  {_fmt_time(info['started_at'])}  {info['name']}")
        print(f"          {detail}")
        if args.verbose:
            counts = info["counts"]
            print(f"          records: {counts}  file: {path}")
    return OK


def cmd_show(args, config: Config) -> int:
    paths = _resolve_inputs(config, args.input, want_all=False)
    if not paths:
        print("No session to show.", file=sys.stderr)
        return FAILED

    path = paths[0]
    records = list(_read_records(path))
    # Subagent work is indented under the session it belongs to.
    nested = {
        r["workflow_id"] for r in records if r.get("type") == "workflow" and r.get("parent_workflow_id") is not None
    }

    tasks = [r for r in records if r.get("type") == "task"]
    if not tasks:
        print(f"{path.name}: no activity recorded yet.")
        return OK

    for record in tasks:
        indent = "    " if record.get("workflow_id") in nested else "  "
        elapsed = ""
        started, ended = record.get("started_at"), record.get("ended_at")
        if isinstance(started, (int, float)) and isinstance(ended, (int, float)):
            elapsed = f"{ended - started:.2f}s"
        print(
            f"{indent}{record.get('subtype', ''):20} {record.get('activity_id', '?'):24} "
            f"{record.get('status', ''):9} {elapsed:>8}"
        )
        if args.verbose and record.get("stderr"):
            print(f"{indent}  ! {record['stderr']}")
    return OK


def cmd_flush(args, config: Config) -> int:
    """Publish buffered records into a running Flowcept backend."""
    try:
        from flowcept.commons.daos.mq_dao.mq_dao_base import MQDao
    except ImportError:
        print("flowcept is not installed; `pip install flowcept-harness[ingest]`", file=sys.stderr)
        return FAILED

    paths = _resolve_inputs(config, args.input, args.all)
    if not paths:
        print("Nothing to flush.", file=sys.stderr)
        return FAILED

    try:
        mq = MQDao.build()
    except Exception as exc:
        print(f"Cannot reach the Flowcept backend: {exc}", file=sys.stderr)
        return FAILED

    total = 0
    for path in paths:
        records = list(_read_records(path))
        if not records:
            continue
        if args.dry_run:
            print(f"would publish {len(records):5} records from {path.name}")
        else:
            mq.bulk_publish(records)
            print(f"published    {len(records):5} records from {path.name}")
        total += len(records)

    if not args.dry_run:
        try:
            # check_safe_stops=False: a flush has no interceptor instance to
            # coordinate, and the control messages it would send carry a None
            # id that crashes the document inserter's bookkeeping.
            mq.stop(check_safe_stops=False)
        except Exception:
            # Best effort: the records are already published.
            pass
        if args.remove:
            for path in paths:
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".lock").unlink(missing_ok=True)

    print(f"{'would publish' if args.dry_run else 'published'} {total} records from {len(paths)} file(s)")
    return OK


def cmd_report(args, config: Config) -> int:
    try:
        from flowcept import Flowcept
    except ImportError:
        print("flowcept is not installed; `pip install flowcept-harness[ingest]`", file=sys.stderr)
        return FAILED

    paths = _resolve_inputs(config, args.input, want_all=False)
    if not paths:
        print("No session to report on.", file=sys.stderr)
        return FAILED

    stats = Flowcept.generate_report(
        report_type=args.type,
        input_jsonl_path=str(paths[0]),
        format=args.format,
        output_path=args.output,
    )
    if args.output:
        print(f"Wrote {args.output} ({stats.get('n_workflows')} workflow(s), {stats.get('n_tasks')} task(s))")
    else:
        print(stats.get("markdown") or stats)
    return OK


def cmd_repair(args, config: Config) -> int:
    """Close sessions whose harness exited without a session-end event."""
    from .recorder import repair_session

    if not config.sessions_dir.is_dir():
        print("No sessions to repair.")
        return OK

    repaired = 0
    for state_path in sorted(config.sessions_dir.glob("*.json")):
        records = repair_session(config, state_path.stem)
        if records:
            repaired += 1
            print(f"repaired {state_path.stem[:8]} ({len(records)} record(s))")
    print(f"{repaired} session(s) repaired.")
    return OK


def cmd_install(args, config: Config) -> int:
    """Print the settings needed to enable capture in a harness."""
    if args.harness == "claude_code":
        print("Add the plugin marketplace, then enable the plugin:\n")
        print("  /plugin marketplace add <path-or-repo>")
        print("  /plugin install flowcept\n")
        print("Or wire the hooks directly in settings.json:\n")
        events = [
            "SessionStart",
            "SessionEnd",
            "UserPromptSubmit",
            "Stop",
            "PreToolUse",
            "PostToolUse",
            "SubagentStart",
            "SubagentStop",
        ]
        hooks = {
            event: [{"hooks": [{"type": "command", "command": f"flowcept-harness hook --event {event}"}]}]
            for event in events
        }
        print(json.dumps({"hooks": hooks}, indent=2))
    else:
        print(f"Set the hook command for {args.harness} to:\n")
        print(f"  flowcept-harness hook --harness {args.harness}")
    return OK


# -- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowcept-harness",
        description="PROV-AGENT provenance capture for AI coding harnesses.",
    )
    parser.add_argument("--version", action="version", version=f"flowcept-harness {__version__}")
    parser.add_argument("--home", help="Override the capture home directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hook", help="Record a hook payload read from stdin.")
    p.add_argument("--harness", default="claude_code")
    p.add_argument("--event", help="Event name, when the payload omits it.")
    p.add_argument("--profile", help="Field-mapping profile for generic harnesses.")
    p.set_defaults(func=cmd_hook)

    p = sub.add_parser("status", help="Show configuration and capture health.")
    p.add_argument("--check-backend", action="store_true", help="Also probe the Flowcept backend.")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("sessions", help="List captured sessions, newest first.")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("show", help="Show the activity of one session.")
    p.add_argument("input", nargs="*", help="Buffer file (default: most recent).")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("flush", help="Publish buffered records to Flowcept.")
    p.add_argument("--input", nargs="*", help="Buffer files (default: most recent).")
    p.add_argument("--all", action="store_true", help="Flush every buffer.")
    p.add_argument("--remove", action="store_true", help="Delete buffers after a successful flush.")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_flush)

    p = sub.add_parser("report", help="Generate a Flowcept report from a buffer.")
    p.add_argument("--input", nargs="*", help="Buffer file (default: most recent).")
    p.add_argument("--type", default="workflow_card")
    p.add_argument("--format", default="markdown")
    p.add_argument("-o", "--output", help="Write to a file instead of stdout.")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("repair", help="Close sessions left open by a crashed harness.")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("install", help="Print the settings that enable capture.")
    p.add_argument("--harness", default="claude_code")
    p.set_defaults(func=cmd_install)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.home:
        os.environ["FLOWCEPT_HARNESS_HOME"] = args.home
    config = load_config()
    return args.func(args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
