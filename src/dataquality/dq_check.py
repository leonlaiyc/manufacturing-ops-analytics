"""
M11 Stage A regression + sanity check (run end to end).

Confirms the event-log schema contract (``schema_contract.py``), the
corruption injectors (``corruptions.py``), and the leakage-safe as-of join
(``asof_join.py``) all behave as documented:

  GATE 1 - clean pass: the default-config event log yields zero violations
           on every one of the six contract clauses (C1-C6).
  GATE 2 - corruption recovery: every injector's output is flagged by
           EXACTLY its expected clause, and the other clean clauses stay
           quiet, for every injector in the registry.
  GATE 3 - reproducibility: two ``validate_log`` reports built from the same
           corrupted frame (same seed) are identical.
  GATE 4 - completeness meta-gate: every contract clause C1-C6 is covered by
           at least one injector exercised in the GATE 2 sweep (fails if a
           clause has no test - guards against the contract growing a clause
           that nothing proves catches anything).
  GATE 5 - as-of join: a tiny synthetic features/labels frame where one
           feature row sits exactly AT the label time and one sits AFTER it;
           ``leakage_safe_asof_join`` must exclude both (only a strictly
           earlier feature row may match). ``audit_join`` must (a) find
           nothing wrong on that safe join, and (b) flag the exact-match row
           on a deliberately leaky ``merge_asof`` (allow_exact_matches=True).

Run:  py src/dataquality/dq_check.py   (exit 0 = all gates pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / "generator"))
sys.path.insert(0, str(SRC / "dataquality"))

from factory_generator import default_config, simulate

from schema_contract import CLAUSES, validate_log
from corruptions import INJECTORS
from asof_join import leakage_safe_asof_join, audit_join


def _report_dict(report: pd.DataFrame) -> dict:
    return dict(zip(report["clause"], report["n_violations"]))


def main() -> int:
    cfg = default_config()  # locked 60-day, seed 42 line
    log, life, meta = simulate(cfg)
    ok = True

    print("=" * 64)
    print("GATE 1 - clean pass (zero violations on every clause)")
    print("=" * 64)
    clean_report = validate_log(log)
    counts = _report_dict(clean_report)
    g1 = all(n == 0 for n in counts.values())
    for clause, n in counts.items():
        print(f"  {clause}: {n} violations")
    print(f"  all clauses zero on clean log : {g1}")
    ok &= g1

    print("=" * 64)
    print("GATE 2 - corruption recovery (injector -> expected clause)")
    print("=" * 64)
    g2 = True
    matrix_rows = []
    for name, (inject_fn, expected_clause) in INJECTORS.items():
        rng = np.random.default_rng(2024)
        corrupted, expected = inject_fn(log, rng)
        assert expected == expected_clause  # sanity: registry consistent with injector
        report = validate_log(corrupted)
        counts = _report_dict(report)
        fired = {c for c, n in counts.items() if n > 0}
        expected_fired = fired == {expected_clause}
        matrix_rows.append((name, expected_clause, sorted(fired)))
        status = "PASS" if expected_fired else "FAIL"
        print(f"  {name:22s} -> expected {expected_clause}, fired {sorted(fired)} : {status}")
        g2 &= expected_fired
    print(f"  every injector flags exactly its expected clause : {g2}")
    ok &= g2

    print("=" * 64)
    print("GATE 3 - reproducibility (same seed, identical report)")
    print("=" * 64)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    corrupted_a, _ = INJECTORS["negate_duration"][0](log, rng_a)
    corrupted_b, _ = INJECTORS["negate_duration"][0](log, rng_b)
    report_a = validate_log(corrupted_a)
    report_b = validate_log(corrupted_b)
    g3 = report_a.drop(columns=["sample"]).equals(report_b.drop(columns=["sample"]))
    print(f"  two reports (seed=99) identical (excl. sample indices) : {g3}")
    ok &= g3

    print("=" * 64)
    print("GATE 4 - completeness meta-gate (every clause has a test)")
    print("=" * 64)
    all_clauses = set(CLAUSES.keys())
    covered_clauses = {expected for _, expected in INJECTORS.values()}
    missing = all_clauses - covered_clauses
    g4 = len(missing) == 0
    print(f"  contract clauses: {sorted(all_clauses)}")
    print(f"  covered by an injector: {sorted(covered_clauses)}")
    print(f"  missing coverage: {sorted(missing)}")
    print(f"  every clause covered by at least one injector : {g4}")
    ok &= g4

    print("=" * 64)
    print("GATE 5 - as-of join (strict earlier-than, no leakage)")
    print("=" * 64)
    label_time = 10.0
    labels_df = pd.DataFrame({"lot_id": [1], "label_time": [label_time], "y": [1]})
    features_df = pd.DataFrame({
        "lot_id": [1, 1, 1],
        "feature_time": [5.0, label_time, label_time + 1.0],  # earlier, exact, after
        "x": [100.0, 200.0, 300.0],
    })

    joined = leakage_safe_asof_join(features_df, labels_df,
                                     feature_time_col="feature_time",
                                     label_time_col="label_time", by="lot_id")
    matched_x = joined["x"].iloc[0]
    g5a = matched_x == 100.0  # only the strictly-earlier row (x=100) may match
    print(f"  safe join matches strictly-earlier feature row (x=100.0) : {matched_x} : {g5a}")

    audit_safe = audit_join(joined, "feature_time", "label_time")
    g5b = len(audit_safe) == 0
    print(f"  audit_join finds 0 leaks on the safe join : {len(audit_safe)} : {g5b}")

    leaky = pd.merge_asof(
        labels_df.sort_values("label_time"), features_df.sort_values("feature_time"),
        left_on="label_time", right_on="feature_time", by="lot_id",
        direction="backward", allow_exact_matches=True,
    )
    leaky_matched_x = leaky["x"].iloc[0]
    audit_leaky = audit_join(leaky, "feature_time", "label_time")
    g5c = len(audit_leaky) == 1
    print(f"  deliberately leaky join matches x={leaky_matched_x} (exact tie)")
    print(f"  audit_join flags the leaky join (1 row) : {len(audit_leaky)} : {g5c}")

    g5 = g5a and g5b and g5c
    ok &= g5

    print("=" * 64)
    print(f"OVERALL: {'ALL GATES PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
