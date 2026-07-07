"""
Regression gates for the M10 Stage B agent loop (``agent_loop.py``).

All gates run offline with the MockLLM: no network access, no API key, no
cost. Run: ``py src/agent/loop_check.py`` (exit 0 on all-pass).

GATE 1  loop mechanics: the mocked session completes, and every tool turn
        sent all of its tool_result blocks back in exactly ONE user message.
GATE 2  traceability: the memo's citation verification is 100 percent with
        at least 3 citations.
GATE 3  fabrication catch: a mock that appends an uncited fabricated number
        to its answer produces a FAILED VERIFICATION session.
GATE 4  reproducibility: two mocked sessions produce identical memos and
        identical run logs (fixed seeds, fixed timestamp).
GATE 5  credential guard: with Anthropic credentials scrubbed from the
        environment, the live adapter raises the clear one-line error that
        names ANTHROPIC_API_KEY (no network attempted, no stack trace).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from agent_loop import (                     # noqa: E402
    AnthropicLLM, AgentCredentialError, MOCK_QUESTION,
    honest_mock, fabricating_mock, run_agent_session,
)


def gate_1_and_2():
    session = run_agent_session(MOCK_QUESTION, honest_mock())
    ok1 = (session["final_text"] != ""
           and session["memo_text"].startswith("# Decision Memo")
           and len(session["turns_meta"]) == 3
           and all(t["n_result_blocks"] == t["n_tool_calls"]
                   and t["n_result_messages"] == 1
                   for t in session["turns_meta"]))
    v = session["verification"]
    ok2 = (session["status"] == "VERIFIED"
           and v["all_found"] and v["total_citations"] >= 3
           and not session["uncited_numbers"])
    return ok1, ok2, session


def gate_3():
    session = run_agent_session(MOCK_QUESTION, fabricating_mock())
    return (session["status"] == "FAILED VERIFICATION"
            and "123456.78" in session["uncited_numbers"]), session


def gate_4(reference):
    again = run_agent_session(MOCK_QUESTION, honest_mock())
    return (again["memo_text"] == reference["memo_text"]
            and again["run_log_jsonl"] == reference["run_log_jsonl"])


def gate_5():
    scrubbed = {}  # empty environment: no key, no auth token
    try:
        AnthropicLLM.check_credentials(scrubbed)
        return False, "no error raised"
    except AgentCredentialError as exc:
        msg = str(exc)
        return ("ANTHROPIC_API_KEY" in msg and "retry" in msg), msg


def main() -> int:
    ok = True

    ok1, ok2, session = gate_1_and_2()
    print(f"GATE 1 loop mechanics: {'PASS' if ok1 else 'FAIL'} "
          f"(turns={len(session['turns_meta'])}, "
          f"single-message results={all(t['n_result_messages'] == 1 for t in session['turns_meta'])})")
    v = session["verification"]
    print(f"GATE 2 traceability: {'PASS' if ok2 else 'FAIL'} "
          f"(citations={v['total_citations']}, all_found={v['all_found']}, "
          f"status={session['status']})")
    ok = ok and ok1 and ok2

    ok3, fab_session = gate_3()
    print(f"GATE 3 fabrication catch: {'PASS' if ok3 else 'FAIL'} "
          f"(status={fab_session['status']}, "
          f"uncited={fab_session['uncited_numbers']})")
    ok = ok and ok3

    ok4 = gate_4(session)
    print(f"GATE 4 reproducibility: {'PASS' if ok4 else 'FAIL'}")
    ok = ok and ok4

    ok5, detail = gate_5()
    print(f"GATE 5 credential guard: {'PASS' if ok5 else 'FAIL'} ({detail})")
    ok = ok and ok5

    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
