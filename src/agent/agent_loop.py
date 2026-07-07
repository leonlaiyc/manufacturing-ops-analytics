"""
LLM agent loop for the M10 agentic decision-support layer (Stage B).

Runs the manual tool-calling loop on top of the Stage A tool layer
(``tools.ToolRegistry``): the model asks for tool calls, this loop executes
them through the logged registry, feeds the results back, and finally builds
a decision memo whose LLM-authored recommendation is verified number by
number against the run log (``run_log.verify_memo_numbers``). A session whose
memo does not verify 100 percent is marked FAILED VERIFICATION, never
silently accepted.

Two LLM backends share the exact same loop code:

- ``AnthropicLLM``: thin adapter over the official ``anthropic`` SDK
  (model ``claude-opus-4-8``, adaptive thinking). Used only when API
  credentials exist in the environment; every check gate below runs without
  it.
- ``MockLLM``: deterministic scripted responses that mimic the SDK response
  surface (``stop_reason``, ``content`` blocks). The mock reads run ids and
  numbers out of the tool results it is fed back (it never hardcodes them),
  so it exercises the same traceability path a live model would.

Verification is two-layered:
1. every ``<number> [run:<id>]`` citation must resolve against the run log
   (Stage A ``verify_memo_numbers``);
2. the LLM-authored section must not contain uncited substantive numbers:
   any number with a decimal point or three or more digits must carry a
   ``[run:<id>]`` tag. Small integers (one or two digits) are allowed uncited
   so ordinary prose ("15 percent demand growth", "3 scenarios") is not
   flagged. The threshold is documented here on purpose: it is a stylized
   guard against fabricated metrics, not a general fact checker.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import ToolRegistry            # noqa: E402
from run_log import RunLogger, verify_memo_numbers  # noqa: E402
from memo import render_memo              # noqa: E402

MODEL_ID = "claude-opus-4-8"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You are a fab operations decision analyst working on a \
validated discrete-event simulation of a synthetic wafer-fab line. You answer \
operational questions (capacity investment, maintenance timing, dispatching \
policy, demand growth) by calling the provided simulation tools and reading \
their results.

Hard rules:
1. You may ONLY cite numbers that appear in tool results you received in this \
conversation. Every number you cite MUST be immediately followed by the run \
tag of the tool call that produced it, in exactly this format: [run:<run_id>] \
where <run_id> is the run_id field of that tool result. Example: "cycle time \
rises by 16.13 [run:ab12cd34ef56ab12] hours".
2. Never invent numbers, tools, or results. If a question cannot be answered \
from the available tools, say so.
3. All results are rankings under stated assumptions from a synthetic, \
fixed-seed simulation. Never present them as real-fab measurements or \
forecasts.
4. Keep the final answer structured: what was measured, what it implies, a \
recommendation, and one sentence on what evidence would change that \
recommendation.
"""


class AgentCredentialError(RuntimeError):
    """Raised when live mode is requested but no API credentials exist."""


CREDENTIAL_MESSAGE = (
    "No Anthropic API credentials found. Set ANTHROPIC_API_KEY (or "
    "ANTHROPIC_AUTH_TOKEN) in the environment and retry."
)


# --------------------------------------------------------------------------- #
# LLM backends
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """Thin adapter over the official anthropic SDK.

    The client is constructed lazily on first ``create`` so that importing
    this module (and running the mocked gates) never requires the SDK to
    resolve credentials. ``check_credentials`` inspects the environment only;
    it is the deterministic guard the check script exercises without any
    network access.
    """

    def __init__(self, model: str = MODEL_ID) -> None:
        self.model = model
        self._client = None

    @staticmethod
    def check_credentials(environ: dict | None = None) -> None:
        import os
        env = environ if environ is not None else os.environ
        if not env.get("ANTHROPIC_API_KEY") and not env.get("ANTHROPIC_AUTH_TOKEN"):
            raise AgentCredentialError(CREDENTIAL_MESSAGE)

    def create(self, system: str, messages: list, tools: list):
        self.check_credentials()
        import anthropic
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()
            except TypeError as exc:
                raise AgentCredentialError(CREDENTIAL_MESSAGE) from exc
        try:
            return self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                messages=messages,
                tools=tools,
            )
        except anthropic.AuthenticationError as exc:
            raise AgentCredentialError(CREDENTIAL_MESSAGE) from exc


@dataclass
class MockToolUseBlock:
    name: str
    input: dict
    id: str
    type: str = "tool_use"


@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockResponse:
    stop_reason: str
    content: list


@dataclass
class MockLLM:
    """Deterministic scripted LLM sharing the loop code with AnthropicLLM.

    ``script`` is a list of steps. A ``("tool", name, args)`` step returns a
    tool_use response; a ``("final", builder)`` step returns an end_turn text
    response, where ``builder(results)`` receives the parsed tool results
    (in call order, each including its run_id) extracted from the tool_result
    blocks previously fed back through ``messages``. The mock therefore reads
    run ids from the conversation, exactly as a live model must.
    """
    script: list
    _step: int = 0
    _counter: int = 0

    def _parsed_results(self, messages: list) -> list[dict]:
        results = []
        for msg in messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    try:
                        results.append(json.loads(block["content"]))
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass
        return results

    def create(self, system: str, messages: list, tools: list) -> MockResponse:
        if self._step >= len(self.script):
            return MockResponse("end_turn", [MockTextBlock("(script exhausted)")])
        step = self.script[self._step]
        self._step += 1
        if step[0] == "tool":
            _, name, args = step
            self._counter += 1
            return MockResponse("tool_use", [
                MockToolUseBlock(name=name, input=dict(args),
                                 id=f"toolu_mock_{self._counter:03d}"),
            ])
        _, builder = step
        return MockResponse("end_turn",
                            [MockTextBlock(builder(self._parsed_results(messages)))])


# --------------------------------------------------------------------------- #
# Uncited-number guard for the LLM-authored section
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
#: A full citation: number, optional short unit, then its [run:<id>] tag.
_CITED_RE = re.compile(r"-?\d[\d,]*\.?\d*\s*[A-Za-z%$]*\s*\[run:[0-9a-fA-F]+\]")
_TAG_RE = re.compile(r"\[run:[0-9a-fA-F]+\]")


def find_uncited_numbers(text: str) -> list[str]:
    """Return substantive numbers in ``text`` that carry no [run:...] tag.

    Substantive = contains a decimal point, or has three or more digits.
    One- and two-digit integers are allowed uncited (prose like "15 percent"
    or "3 scenarios"). Complete citations (and any bare run tags, whose hex
    ids can contain long digit runs) are stripped before scanning, so only
    genuinely untagged numbers are flagged.
    """
    cleaned = _CITED_RE.sub(" ", text)
    cleaned = _TAG_RE.sub(" ", cleaned)
    flagged = []
    for m in _NUMBER_RE.finditer(cleaned):
        token = m.group(0)
        digits = token.replace("-", "").replace(",", "").replace(".", "")
        if "." not in token and len(digits) < 3:
            continue
        flagged.append(token)
    return flagged


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run_agent_session(question: str, llm, registry: ToolRegistry | None = None,
                      logger: RunLogger | None = None, max_turns: int = 12,
                      timestamp: str = "2026-07-07T00:00:00+00:00") -> dict:
    """Run one agent session: question in, verified decision memo out.

    Returns a dict with question, turns_meta (per-turn tool call accounting),
    final_text (the LLM's answer), memo_text, verification (the Stage A
    citation report), uncited_numbers (guard layer 2), status
    ("VERIFIED" or "FAILED VERIFICATION"), and the run-log entries.
    """
    registry = registry if registry is not None else ToolRegistry()
    logger = logger if logger is not None else RunLogger()
    tools = registry.schemas()
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_results: list[dict] = []
    turns_meta: list[dict] = []
    final_text = ""

    for _turn in range(max_turns):
        response = llm.create(SYSTEM_PROMPT, messages, tools)
        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content
                           if getattr(b, "type", None) == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})
            result_blocks = []
            for block in tool_blocks:
                try:
                    result = registry.call(block.name, dict(block.input),
                                           logger=logger, timestamp=timestamp)
                    result = dict(result)
                    result["_tool_name"] = block.name
                    tool_results.append(result)
                    payload = {k: v for k, v in result.items() if k != "_tool_name"}
                    result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(payload, sort_keys=True, default=str),
                    })
                except (ValueError, KeyError) as exc:
                    result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool error: {exc}",
                        "is_error": True,
                    })
            # All results for the turn go back in ONE user message.
            messages.append({"role": "user", "content": result_blocks})
            turns_meta.append({"n_tool_calls": len(tool_blocks),
                               "n_result_blocks": len(result_blocks),
                               "n_result_messages": 1})
        else:
            final_text = "".join(getattr(b, "text", "") for b in response.content
                                 if getattr(b, "type", None) == "text")
            break

    memo_text = render_memo(question, tool_results, recommendation_stub=final_text)
    verification = verify_memo_numbers(memo_text, logger)
    uncited = find_uncited_numbers(final_text)
    verified = (verification["total_citations"] >= 1
                and verification["all_found"] and not uncited)
    return {
        "question": question,
        "model": getattr(llm, "model", type(llm).__name__),
        "turns_meta": turns_meta,
        "final_text": final_text,
        "memo_text": memo_text,
        "verification": verification,
        "uncited_numbers": uncited,
        "status": "VERIFIED" if verified else "FAILED VERIFICATION",
        "run_log_jsonl": logger.to_jsonl(),
    }


# --------------------------------------------------------------------------- #
# Canned mock scripts (used by loop_check.py and as live-session reference)
# --------------------------------------------------------------------------- #
MOCK_QUESTION = "Is one more litho tool worth it under 15 percent demand growth?"

_MOCK_CALLS = [
    ("tool", "get_kpi_baseline", {}),
    ("tool", "run_demand_whatif", {"demand_factor": 1.15, "n_reps": 8, "seed0": 5000}),
    ("tool", "run_capacity_whatif", {"station": "LITHO",
                                     "demand_factors": [1.0, 1.15],
                                     "n_reps": 8, "seed0": 1000}),
]


def _honest_final(results: list[dict]) -> str:
    base, demand, cap = results[0], results[1], results[2]
    d_ct = demand["summary"][0]["d_cycle_time"]
    rows = {row["factor"]: row for row in cap["per_factor"]}
    return (
        "Measured effects: demand growth of 15 percent alone moves mean cycle "
        f"time by {d_ct:.4f} [run:{demand['run_id']}] hours versus the "
        f"baseline of {base['mean_cycle_time_hours']:.4f} [run:{base['run_id']}] "
        "hours. Adding one LITHO tool changes throughput by "
        f"{rows[1.0]['mean_d_throughput']:.4f} [run:{cap['run_id']}] lots per "
        f"hour at baseline demand and {rows[1.15]['mean_d_throughput']:.4f} "
        f"[run:{cap['run_id']}] lots per hour under the higher demand. "
        "Recommendation: the extra LITHO tool matters more as demand grows; "
        "weigh the measured gain against tool cost before committing. This "
        "recommendation would change if the capacity delta at higher demand "
        "were measured near zero, or if the cost model ranked another station "
        "higher."
    )


def _fabricating_final(results: list[dict]) -> str:
    honest = _honest_final(results)
    return honest + (" Additionally this change saves 123456.78 dollars per "
                     "quarter in congestion cost.")


def honest_mock() -> MockLLM:
    return MockLLM(script=[*_MOCK_CALLS, ("final", _honest_final)])


def fabricating_mock() -> MockLLM:
    return MockLLM(script=[*_MOCK_CALLS, ("final", _fabricating_final)])
