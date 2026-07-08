"""
Run logger for the M10 agentic decision-support layer (Stage A).

Every tool invocation (``src/agent/tools.py``) appends exactly one JSON line
to a session log. The guarantee this module exists to make mechanical, not
aspirational: "every number in a memo traces to a logged simulation run."

Design
------
- ``run_id`` is a deterministic hash of (tool name, canonicalized args, seed).
  Canonicalization = ``json.dumps(args, sort_keys=True, default=str)`` so
  argument order and value types (e.g. tuple vs list) never change the id,
  only the actual values do. Same tool + same args + same seed always yields
  the same id (GATE 5 in ``agent_check.py``); changing any argument changes
  the hash with overwhelming probability (SHA-256, truncated for readability).
- The log is plain JSON Lines (one ``json.dumps(...)`` object per line) so it
  is diffable, appendable, and readable without a special parser.
- ``verify_memo_numbers`` is the traceability check: it scans memo text for
  citations of the form ``<number> [run:<run_id>]``, looks up each cited
  run's logged result, and confirms the number actually appears somewhere in
  that run's numeric result (within a small formatting tolerance, since the
  memo prints rounded numbers but the log stores full precision floats).

No LLM anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonicalize_args(args: dict) -> str:
    """Stable JSON string for a dict of arguments, independent of key order.

    ``default=str`` covers argument values that are not natively JSON
    serializable (e.g. a tuple passed through unchanged); this only affects
    the string used for hashing, never the args stored in the log entry.
    """
    return json.dumps(args, sort_keys=True, default=str)


def compute_run_id(tool_name: str, args: dict, seed: int) -> str:
    """Deterministic run id: sha256(tool_name + canonical_args + seed), truncated.

    16 hex characters (64 bits) is ample to avoid accidental collision across
    a single session's tool calls while staying short enough to read and cite
    in a memo as ``[run:<run_id>]``.
    """
    payload = f"{tool_name}|{canonicalize_args(args)}|{seed}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class RunLogger:
    """Appends one JSON line per tool invocation; can replay/read the log back.

    ``entries`` mirrors the on-disk log in memory (in append order) so a
    session can be built and verified without a filesystem round trip, and
    also written to ``log_path`` if one is given.
    """
    log_path: Path | None = None
    entries: list[dict] = field(default_factory=list)

    def log(self, run_id: str, tool: str, args: dict, seed: int,
             engine_config: dict, result: dict, timestamp: str) -> dict:
        """Append one run record and return it.

        ``timestamp`` is passed in (not read from the system clock here) so
        callers that need reproducible logs (the replay harness, GATE 1) can
        supply a fixed value; live callers can pass
        ``datetime.now(timezone.utc).isoformat()``.
        """
        entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "tool": tool,
            "args": args,
            "seed": seed,
            "engine_config": engine_config,
            "result": result,
        }
        self.entries.append(entry)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def by_run_id(self) -> dict[str, dict]:
        """Map run_id -> entry, last write wins (a run_id should be unique)."""
        return {e["run_id"]: e for e in self.entries}

    def to_jsonl(self) -> str:
        """Serialize the whole in-memory session log as JSON Lines text."""
        return "\n".join(json.dumps(e, sort_keys=True) for e in self.entries) + (
            "\n" if self.entries else "")

    @classmethod
    def from_jsonl(cls, text: str) -> "RunLogger":
        """Rebuild a RunLogger (no log_path, in-memory only) from JSONL text."""
        logger = cls()
        for line in text.splitlines():
            line = line.strip()
            if line:
                logger.entries.append(json.loads(line))
        return logger


#: Matches a citation like "14.90 [run:ab12cd34ef56ab12]" or "-2.87h [run:...]".
#: Captures (number-as-written, run_id). The number group allows optional sign,
#: digits, optional decimal part - it deliberately does NOT try to capture a
#: trailing unit suffix (h, %, $) so callers can format numbers however they
#: like in memo prose; the tolerance check below strips non-numeric trailing
#: characters before comparing.
_CITATION_RE = re.compile(
    r"(-?\d[\d,]*\.?\d*)\s*[A-Za-z%$]*\s*\[run:([0-9a-fA-F]+)\]"
)


def _extract_citations(memo_text: str) -> list[tuple[str, str]]:
    """Return [(number_as_written, run_id), ...] in the order they appear."""
    return [(m.group(1), m.group(2)) for m in _CITATION_RE.finditer(memo_text)]


def _flatten_numbers(obj: Any) -> list[float]:
    """Recursively collect every int/float leaf from a JSON-like structure."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out  # bool is a subclass of int in Python; not a cited number
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_numbers(v))
    return out


def _parse_cited_number(text: str) -> float:
    return float(text.replace(",", ""))


def verify_memo_numbers(memo_text: str, session_log: RunLogger,
                         rel_tol: float = 1e-3, abs_tol: float = 1e-6
                         ) -> dict:
    """Check every ``<number> [run:<id>]`` citation in ``memo_text`` against
    the referenced run's logged result.

    A citation PASSES when its run_id is found in ``session_log`` AND the
    cited number matches at least one numeric leaf of that run's logged
    ``result`` within tolerance (formatting tolerance: the memo prints
    rounded numbers, e.g. "14.90" for a logged 14.8963..., so comparison
    uses both a relative and an absolute tolerance and also checks against
    the value rounded to the same number of decimal places the citation
    used, so exact-match roundings always pass).

    Returns
    -------
    dict with keys:
      total_citations : int
      all_found        : bool  (True iff every citation passed)
      report           : list[dict], one per citation, in citation order:
                         {citation_text, run_id, cited_value, run_found,
                          value_found, matched_against}
    """
    by_id = session_log.by_run_id()
    citations = _extract_citations(memo_text)
    report = []
    for raw_number, run_id in citations:
        cited_value = _parse_cited_number(raw_number)
        entry = by_id.get(run_id)
        run_found = entry is not None
        value_found = False
        matched_against = None
        if run_found:
            candidates = _flatten_numbers(entry["result"])
            # number of decimals in the citation, for a rounding-exact check
            decimals = len(raw_number.split(".")[1]) if "." in raw_number else 0
            for cand in candidates:
                if round(cand, decimals) == cited_value:
                    value_found = True
                    matched_against = cand
                    break
                diff = abs(cand - cited_value)
                if diff <= abs_tol or diff <= rel_tol * max(abs(cand), abs(cited_value), 1.0):
                    value_found = True
                    matched_against = cand
                    break
        report.append({
            "citation_text": f"{raw_number} [run:{run_id}]",
            "run_id": run_id,
            "cited_value": cited_value,
            "run_found": run_found,
            "value_found": value_found,
            "matched_against": matched_against,
        })
    all_found = bool(report) and all(r["run_found"] and r["value_found"] for r in report)
    return {
        "total_citations": len(report),
        "all_found": all_found,
        "report": report,
    }
