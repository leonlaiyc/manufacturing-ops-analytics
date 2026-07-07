"""
Run one LIVE agent session against the Anthropic API and save the evidence.

Usage (requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in the environment):

    py scripts/run_live_agent_session.py
    py scripts/run_live_agent_session.py --question "When should we schedule PM on LITHO?"

Outputs, under reports/agent_sessions/session_<UTC timestamp>/:
    transcript.json  question, model, per-run tool log, final answer,
                     citation verification report, status
    memo.md          the rendered decision memo (markdown)

Exit codes: 0 = session VERIFIED; 1 = missing credentials; 2 = session ran
but FAILED VERIFICATION (the memo is still saved, flagged, for inspection).

This script performs exactly one live session per invocation and never runs
from the check gates (those use the MockLLM; see src/agent/loop_check.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "agent"))

from agent_loop import (                     # noqa: E402
    AnthropicLLM, AgentCredentialError, MOCK_QUESTION, run_agent_session,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=MOCK_QUESTION,
                        help="Operational question for the agent.")
    args = parser.parse_args()

    try:
        AnthropicLLM.check_credentials()
    except AgentCredentialError as exc:
        print(str(exc))
        return 1

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "reports" / "agent_sessions" / f"session_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running live agent session ({stamp}) ...")
    print(f"Question: {args.question}")
    session = run_agent_session(args.question, AnthropicLLM(),
                                timestamp=now.isoformat())

    transcript = {k: session[k] for k in
                  ("question", "model", "turns_meta", "final_text",
                   "verification", "uncited_numbers", "status")}
    transcript["run_log"] = [json.loads(line) for line in
                             session["run_log_jsonl"].splitlines() if line]
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "memo.md").write_text(session["memo_text"], encoding="utf-8")

    v = session["verification"]
    print(f"Tool runs logged: {len(transcript['run_log'])}")
    print(f"Citations: {v['total_citations']}, all_found={v['all_found']}, "
          f"uncited={session['uncited_numbers']}")
    print(f"Status: {session['status']}")
    print(f"Saved: {out_dir}")
    return 0 if session["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    sys.exit(main())
