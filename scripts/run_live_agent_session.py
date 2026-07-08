"""
Run one LIVE agent session against an external LLM API and save the evidence.

Usage (defaults to OpenAI, requires OPENAI_API_KEY in the environment):

    py scripts/run_live_agent_session.py
    py scripts/run_live_agent_session.py --provider openai
    py scripts/run_live_agent_session.py --provider anthropic
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
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "agent"))

from agent_loop import (                     # noqa: E402
    AnthropicLLM, AgentCredentialError, MOCK_QUESTION, run_agent_session,
)


OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_CREDENTIAL_MESSAGE = (
    "No OpenAI API credentials found. Set OPENAI_API_KEY in the environment "
    "and retry."
)


@dataclass
class OpenAIToolUseBlock:
    name: str
    input: dict
    id: str
    type: str = "tool_use"


@dataclass
class OpenAITextBlock:
    text: str
    type: str = "text"


@dataclass
class OpenAIResponse:
    stop_reason: str
    content: list


class OpenAILLM:
    """Small Chat Completions adapter for the existing agent loop.

    The project loop expects an Anthropic-shaped response surface:
    ``tool_use`` blocks while tools are needed and text blocks for the final
    answer. This adapter keeps that internal contract while using OpenAI's
    external tool-calling API.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", OPENAI_MODEL)

    @staticmethod
    def check_credentials(environ: dict | None = None) -> None:
        env = environ if environ is not None else os.environ
        if not env.get("OPENAI_API_KEY"):
            raise AgentCredentialError(OPENAI_CREDENTIAL_MESSAGE)

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _convert_messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            if role == "assistant" and isinstance(content, list):
                tool_calls = []
                text_parts = []
                for block in content:
                    btype = getattr(block, "type", None)
                    if btype == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input, sort_keys=True),
                            },
                        })
                    elif btype == "text":
                        text_parts.append(getattr(block, "text", ""))
                if tool_calls:
                    out.append({
                        "role": "assistant",
                        "content": "".join(text_parts) or None,
                        "tool_calls": tool_calls,
                    })
                elif text_parts:
                    out.append({"role": "assistant", "content": "".join(text_parts)})
                continue
            if role == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        })
        return out

    def create(self, system: str, messages: list, tools: list) -> OpenAIResponse:
        self.check_credentials()
        payload = {
            "model": self.model,
            "messages": self._convert_messages(system, messages),
            "tools": self._convert_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_tokens": 1400,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (401, 403):
                raise AgentCredentialError(OPENAI_CREDENTIAL_MESSAGE) from exc
            raise RuntimeError(f"OpenAI API request failed ({exc.code}): {body}") from exc

        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            blocks = []
            for call in tool_calls:
                function = call.get("function", {})
                raw_args = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
                blocks.append(OpenAIToolUseBlock(
                    name=function.get("name", ""),
                    input=args,
                    id=call["id"],
                ))
            return OpenAIResponse("tool_use", blocks)
        return OpenAIResponse("end_turn", [OpenAITextBlock(message.get("content") or "")])


def _build_llm(provider: str):
    if provider == "anthropic":
        AnthropicLLM.check_credentials()
        return AnthropicLLM()
    OpenAILLM.check_credentials()
    return OpenAILLM()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "anthropic"),
                        default=os.environ.get("LIVE_AGENT_PROVIDER", "openai"),
                        help="External LLM provider for the live session.")
    parser.add_argument("--question", default=MOCK_QUESTION,
                        help="Operational question for the agent.")
    args = parser.parse_args()

    try:
        llm = _build_llm(args.provider)
    except AgentCredentialError as exc:
        print(str(exc))
        return 1

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "reports" / "agent_sessions" / f"session_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running live agent session ({stamp}) ...")
    print(f"Provider: {args.provider}")
    print(f"Model: {llm.model}")
    print(f"Question: {args.question}")
    session = run_agent_session(args.question, llm, timestamp=now.isoformat())

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
