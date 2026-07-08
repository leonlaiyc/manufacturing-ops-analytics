"""
M11 Stage A - event-log schema contract.

The contract is expressed as plain data (column lists, dtype expectations,
the locked route) plus a set of named "clause" functions, one per family of
corruption the contract is meant to catch. Each clause takes the event-log
DataFrame and returns a "violation DataFrame": a slice of the offending rows
(empty if the clause is satisfied), so a caller can both count violations and
inspect examples. ``validate_log`` runs every clause and returns a tidy report.

Locked event-log schema (see CLAUDE.md):
  lot_id, product_type, step_seq, station, queue_entry_time,
  process_start_time, process_complete_time, tool_id

Locked route (see CLAUDE.md / factory_generator.default_config): CLEAN,
FURNACE, DEPO, LITHO, ETCH, LITHO, IMPLANT, METRO (LITHO re-entrant, visited
at route positions 3 and 5, zero-indexed). One row per (lot, route position);
step_seq is the zero-indexed route position, so the sequence of step_seq
values for a clean lot is exactly ``0, 1, ..., len(route) - 1`` in order.

Clause list:
  C1 required columns present with expected dtypes (times numeric, ids not
     float/time-like).
  C2 no missing values in required columns.
  C3 time sanity: queue_entry_time <= process_start_time <=
     process_complete_time, all finite, all >= 0 (no negative durations).
  C4 route validity: per lot, the (step_seq, station) sequence is exactly a
     prefix-or-complete traversal of the locked route in order (catches
     impossible hops and wrong stations); step_seq strictly increasing.
  C5 no duplicate (lot_id, step_seq) rows.
  C6 tool consistency: tool_id's station prefix matches the row's station
     (e.g. LITHO-1 only appears on a LITHO row).

Design note: C1 is a structural precondition for every later clause (they all
assume the required columns exist and, where relevant, are numeric). When C1
fails on a required column, ``validate_log`` skips clauses that would raise on
the malformed column rather than reporting a wall of derived failures; the
report still surfaces C1's own violation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "lot_id", "product_type", "step_seq", "station",
    "queue_entry_time", "process_start_time", "process_complete_time",
    "tool_id",
)

#: Columns whose values must be numeric (times) vs identifier-like (ids /
#: labels, i.e. NOT float and NOT datetime - lot_id is an integer index in
#: the generator's own output, so "string dtype" is not required, only
#: "not a floating-point/time dtype").
NUMERIC_TIME_COLUMNS = ("queue_entry_time", "process_start_time", "process_complete_time")
IDENTIFIER_COLUMNS = ("lot_id", "product_type", "step_seq", "station", "tool_id")

#: Locked route (see CLAUDE.md). LITHO appears twice (re-entrant).
LOCKED_ROUTE = ("CLEAN", "FURNACE", "DEPO", "LITHO", "ETCH", "LITHO", "IMPLANT", "METRO")

TIME_COLUMN_ORDER = ("queue_entry_time", "process_start_time", "process_complete_time")


def _empty_violations(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0]


def check_c1_columns_and_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """C1: required columns present, with expected dtype family.

    Returns a one-row-per-problem DataFrame (columns: column, issue) rather
    than a slice of the input log, since a missing column has no input rows
    to point at.
    """
    problems = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            problems.append({"column": col, "issue": "missing required column"})
            continue
        dtype = df[col].dtype
        if col in NUMERIC_TIME_COLUMNS:
            if not pd.api.types.is_numeric_dtype(dtype):
                problems.append({"column": col, "issue": f"expected numeric dtype, got {dtype}"})
        elif col in IDENTIFIER_COLUMNS:
            if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype):
                problems.append({"column": col, "issue": f"expected identifier-like dtype, got {dtype}"})
    return pd.DataFrame(problems, columns=["column", "issue"])


def check_c2_no_missing(df: pd.DataFrame) -> pd.DataFrame:
    """C2: no missing values (NaN/NaT/None) in any required column present."""
    cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if not cols:
        return _empty_violations(df)
    mask = df[cols].isna().any(axis=1)
    return df.loc[mask]


def check_c3_time_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """C3: queue_entry_time <= process_start_time <= process_complete_time,
    all three finite, all three >= 0 (no negative durations).

    Assumes C1 (numeric dtype) and C2 (no missing values) already hold for
    these columns: rows with a missing value in any of the three columns are
    C2's responsibility and are skipped here (a NaN comparison is always
    False, which would otherwise make C3 fire redundantly for the same row
    C2 already flags). If a column has been corrupted to a non-numeric
    dtype, this clause defers to C1 entirely and reports no violations.
    """
    cols = list(TIME_COLUMN_ORDER)
    if any(c not in df.columns for c in cols):
        return _empty_violations(df)
    if not all(pd.api.types.is_numeric_dtype(df[c].dtype) for c in cols):
        return _empty_violations(df)
    t = df[cols]
    has_value = t.notna().all(axis=1)
    t_num = t.to_numpy(dtype="float64")
    finite = np.isfinite(t_num)
    all_finite = finite.all(axis=1)
    non_negative = np.where(np.isnan(t_num), True, t_num >= 0).all(axis=1)
    ordered = (t["queue_entry_time"] <= t["process_start_time"]) & \
              (t["process_start_time"] <= t["process_complete_time"])
    ok = ~has_value | (all_finite & non_negative & ordered)
    return df.loc[~ok]


def check_c4_route_validity(df: pd.DataFrame, route: tuple = LOCKED_ROUTE) -> pd.DataFrame:
    """C4: per lot, (step_seq, station) is a prefix-or-complete traversal of
    ``route`` in order: the DISTINCT step_seq values for a lot, sorted, must
    equal 0..k-1 for some k <= len(route) (strictly increasing, no gaps),
    and station at each step_seq must equal route[step_seq]. Also catches
    step_seq values outside [0, len(route)-1] (impossible hop off the end of
    the route) and non-integer step_seq.

    Exact duplicate (lot_id, step_seq) rows are deliberately NOT this
    clause's concern (C5 covers those separately): the prefix shape is
    checked on distinct step_seq values so a duplicated row does not also
    trip C4 for the same underlying problem C5 already reports.
    """
    needed = {"lot_id", "step_seq", "station"}
    if not needed.issubset(df.columns):
        return _empty_violations(df)

    bad_idx = []
    n_steps = len(route)
    for lot_id, g in df.groupby("lot_id", sort=False):
        steps = g["step_seq"]
        # non-integer or out-of-range step_seq -> whole lot's rows flagged
        is_int_like = steps.apply(lambda v: pd.notna(v) and float(v) == int(v))
        in_range = steps.between(0, n_steps - 1)
        if not (is_int_like & in_range).all():
            bad_idx.extend(g.index.tolist())
            continue
        # dedupe on (step_seq, station) so an exact duplicate row does not
        # also break the prefix-shape check (that is C5's job)
        distinct = g.drop_duplicates(subset=["step_seq", "station"]).sort_values("step_seq")
        seq = distinct["step_seq"].astype(int).tolist()
        expected_prefix = list(range(len(seq)))
        if seq != expected_prefix:
            # not strictly increasing 0..k-1 (gap or disorder in distinct steps)
            bad_idx.extend(g.index.tolist())
            continue
        wrong_station = distinct["station"].to_numpy() != np.array(
            [route[s] for s in seq]
        )
        if wrong_station.any():
            bad_steps = set(distinct.loc[distinct.index[wrong_station], "step_seq"])
            bad_idx.extend(g.index[g["step_seq"].isin(bad_steps)].tolist())

    if not bad_idx:
        return _empty_violations(df)
    return df.loc[sorted(set(bad_idx))]


def check_c5_no_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """C5: no duplicate (lot_id, step_seq) rows."""
    needed = {"lot_id", "step_seq"}
    if not needed.issubset(df.columns):
        return _empty_violations(df)
    mask = df.duplicated(subset=["lot_id", "step_seq"], keep=False)
    return df.loc[mask]


def check_c6_tool_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """C6: tool_id's station prefix matches the row's station, e.g. LITHO-1
    is only valid on a LITHO row. tool_id is expected to be "{station}-{n}".
    """
    needed = {"station", "tool_id"}
    if not needed.issubset(df.columns):
        return _empty_violations(df)
    tool_station = df["tool_id"].astype(str).str.rsplit("-", n=1).str[0]
    mask = tool_station != df["station"].astype(str)
    return df.loc[mask]


#: Ordered clause registry: name -> callable(df) -> violation DataFrame.
CLAUSES: dict = {
    "C1": check_c1_columns_and_dtypes,
    "C2": check_c2_no_missing,
    "C3": check_c3_time_sanity,
    "C4": check_c4_route_validity,
    "C5": check_c5_no_duplicates,
    "C6": check_c6_tool_consistency,
}

CLAUSE_DESCRIPTIONS = {
    "C1": "required columns present with expected dtypes",
    "C2": "no missing values in required columns",
    "C3": "time sanity (order, finite, non-negative)",
    "C4": "route validity (prefix-or-complete traversal, correct stations)",
    "C5": "no duplicate (lot_id, step_seq) rows",
    "C6": "tool_id station prefix matches row station",
}


def validate_log(df: pd.DataFrame, route: tuple = LOCKED_ROUTE, sample_n: int = 5) -> pd.DataFrame:
    """Run every contract clause against ``df``; return a tidy report.

    One row per clause: clause, description, n_violations, sample (a list of
    up to ``sample_n`` violating row indices, for quick inspection). If C1
    reports a missing/mistyped required column, clauses that read that
    column are skipped for THIS run (their own defensive checks already
    return empty on a missing column; a dtype problem such as a numeric
    column that got parsed as string still lets comparisons run, so C3/C4
    are attempted regardless and simply may not fire).
    """
    rows = []
    for name, fn in CLAUSES.items():
        if name == "C1":
            viol = fn(df)
        else:
            viol = fn(df, route) if name == "C4" else fn(df)
        n = len(viol)
        sample = viol.head(sample_n).index.tolist() if n else []
        rows.append({
            "clause": name,
            "description": CLAUSE_DESCRIPTIONS[name],
            "n_violations": n,
            "sample": sample,
        })
    return pd.DataFrame(rows, columns=["clause", "description", "n_violations", "sample"])
