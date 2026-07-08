"""
M11 Stage A - seeded corruption injectors.

One injector per contract-clause family (see ``schema_contract.py``). Each
injector takes a clean event log plus a ``numpy.random.Generator`` and
returns ``(corrupted_df, expected_clause)`` so a gate script can assert that
running ``validate_log`` on the corrupted frame flags exactly the expected
clause (and, ideally, only that clause). This is how the contract's claims
are checked rather than merely asserted: each injector is a small, documented
proof that its target clause actually catches the corruption it names.

Every injector is non-destructive to its input (operates on a copy) and
touches only a handful of rows/cells, chosen via the supplied RNG, so results
are reproducible given the same seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema_contract import LOCKED_ROUTE, NUMERIC_TIME_COLUMNS


def _pick_rows(df: pd.DataFrame, rng: np.random.Generator, n: int) -> np.ndarray:
    n = min(n, len(df))
    return rng.choice(df.index.to_numpy(), size=n, replace=False)


def stringify_timestamps(df: pd.DataFrame, rng: np.random.Generator) -> tuple:
    """C1 family: cast a numeric time column to string dtype, breaking the
    "times are numeric" dtype expectation for the whole column.

    Expected clause: C1 (required columns present with expected dtypes).
    """
    out = df.copy()
    col = rng.choice(list(NUMERIC_TIME_COLUMNS))
    out[col] = out[col].astype(str)
    return out, "C1"


def drop_timestamps(df: pd.DataFrame, rng: np.random.Generator, n: int = 5) -> tuple:
    """C2 family: null out ``process_start_time`` on a handful of rows.

    Expected clause: C2 (missing values in required columns).
    """
    out = df.copy()
    idx = _pick_rows(out, rng, n)
    out.loc[idx, "process_start_time"] = np.nan
    return out, "C2"


def negate_duration(df: pd.DataFrame, rng: np.random.Generator, n: int = 5) -> tuple:
    """C3 family: push ``process_complete_time`` before ``process_start_time``
    on a handful of rows, producing a negative-duration operation.

    Expected clause: C3 (time sanity / ordering).
    """
    out = df.copy()
    idx = _pick_rows(out, rng, n)
    out.loc[idx, "process_complete_time"] = out.loc[idx, "process_start_time"] - 1.0
    return out, "C3"


def duplicate_rows(df: pd.DataFrame, rng: np.random.Generator, n: int = 5) -> tuple:
    """C5 family: duplicate a handful of existing (lot_id, step_seq) rows.

    Expected clause: C5 (no duplicate (lot_id, step_seq) rows).
    """
    idx = _pick_rows(df, rng, n)
    extra = df.loc[idx]
    out = pd.concat([df, extra], ignore_index=True)
    return out, "C5"


def impossible_hop(df: pd.DataFrame, rng: np.random.Generator, n: int = 5) -> tuple:
    """C4 family: swap a row's ``station`` to one that is not the route
    station for its ``step_seq`` (an impossible hop off the locked route).

    Expected clause: C4 (route validity).

    ``tool_id`` is rewritten to match the new (wrong) station so the row
    stays internally consistent (C6 tool-consistency is satisfied); the row
    is isolated as a pure route-validity violation, not a compound one.
    """
    out = df.copy()
    idx = _pick_rows(out, rng, n)
    stations = sorted(set(LOCKED_ROUTE))
    for i in idx:
        current = out.at[i, "station"]
        choices = [s for s in stations if s != current]
        new_station = rng.choice(choices)
        out.at[i, "station"] = new_station
        old_tool_id = str(out.at[i, "tool_id"])
        tool_suffix = old_tool_id.rsplit("-", 1)[-1]
        out.at[i, "tool_id"] = f"{new_station}-{tool_suffix}"
    return out, "C4"


def orphan_tool(df: pd.DataFrame, rng: np.random.Generator, n: int = 5) -> tuple:
    """C6 family: assign a ``tool_id`` from a different station than the
    row's own station (an orphaned / mismatched tool id).

    Expected clause: C6 (tool_id station prefix matches row station).
    """
    out = df.copy()
    idx = _pick_rows(out, rng, n)
    stations = sorted(set(LOCKED_ROUTE))
    for i in idx:
        current_station = out.at[i, "station"]
        other = rng.choice([s for s in stations if s != current_station])
        out.at[i, "tool_id"] = f"{other}-1"
    return out, "C6"


def shuffle_step_seq(df: pd.DataFrame, rng: np.random.Generator, n_lots: int = 3) -> tuple:
    """C4 family: shuffle the ``step_seq`` values within a handful of lots,
    breaking strict-increasing route order without changing which stations
    the lot visited.

    Expected clause: C4 (route validity; step_seq strictly increasing per lot).
    """
    out = df.copy()
    lot_ids = out["lot_id"].unique()
    n_lots = min(n_lots, len(lot_ids))
    chosen_lots = rng.choice(lot_ids, size=n_lots, replace=False)
    for lot in chosen_lots:
        mask = out["lot_id"] == lot
        rows = out.index[mask]
        if len(rows) < 2:
            continue
        shuffled = rng.permutation(out.loc[rows, "step_seq"].to_numpy())
        out.loc[rows, "step_seq"] = shuffled
    return out, "C4"


#: Registry: injector name -> (callable, expected clause). Used by
#: ``dq_check.py`` GATE 2 (recovery sweep) and GATE 4 (completeness meta-gate).
INJECTORS: dict = {
    "stringify_timestamps": (stringify_timestamps, "C1"),
    "drop_timestamps": (drop_timestamps, "C2"),
    "negate_duration": (negate_duration, "C3"),
    "duplicate_rows": (duplicate_rows, "C5"),
    "impossible_hop": (impossible_hop, "C4"),
    "orphan_tool": (orphan_tool, "C6"),
    "shuffle_step_seq": (shuffle_step_seq, "C4"),
}
