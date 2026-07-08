"""
M11 - event-log schema contract, corruption injectors, leakage-safe as-of
join helper (Stage A), plus drift monitoring and conformal prediction
intervals for the M7 virtual-metrology model (Stage B).

Stage A:
  - ``schema_contract.py`` : the contract as data (required columns, dtype
                             expectations, locked route) plus six named
                             clauses (C1-C6), each returning a violation
                             DataFrame, and ``validate_log`` which runs all
                             of them and returns a tidy report.
  - ``corruptions.py``     : seeded injectors, one per clause family, each
                             returning ``(corrupted_df, expected_clause)`` so
                             a gate script can prove the matching clause
                             actually fires.
  - ``asof_join.py``       : ``leakage_safe_asof_join`` (strict "less than",
                             not "less than or equal", feature-to-label join)
                             and ``audit_join`` (belt-and-braces post-hoc
                             leakage check for any already-joined frame).
  - ``dq_check.py``        : gate script (clean pass, corruption recovery,
                             reproducibility, completeness meta-gate, as-of
                             join correctness).

Stage B:
  - ``drift.py``           : hand-built rolling-window drift monitor
                             (reference window vs sliding test window,
                             standardized mean-difference score, k-consecutive
                             alarm rule). Two channels: a station's daily mean
                             process time and the daily arrival count, plus a
                             stylized arrival-rate cut-over builder.
  - ``conformal.py``       : split conformal prediction intervals wrapping
                             the M7 virtual-metrology OLS model (train /
                             calibration / test three-way time split,
                             finite-sample-correct quantile of calibration
                             residuals).
  - ``reliability_check.py``: gate script (drift recovery, arrival-shift
                             recovery, conformal coverage, noise sensitivity,
                             reproducibility, model-card meta-gate).

Import convention (matches the rest of ``src/``): modules use BARE imports
(``from schema_contract import ...``) and consumers put ``src/dataquality``
on ``sys.path`` -- see ``dq_check.py`` / ``reliability_check.py`` for the
pattern.

Public exports by module:

  schema_contract : REQUIRED_COLUMNS, NUMERIC_TIME_COLUMNS,
                    IDENTIFIER_COLUMNS, LOCKED_ROUTE, CLAUSES,
                    CLAUSE_DESCRIPTIONS, validate_log,
                    check_c1_columns_and_dtypes .. check_c6_tool_consistency
  corruptions     : INJECTORS, stringify_timestamps, drop_timestamps,
                    negate_duration, duplicate_rows, impossible_hop,
                    orphan_tool, shuffle_step_seq
  asof_join       : leakage_safe_asof_join, audit_join
  drift           : daily_mean_process_time, daily_arrival_count,
                    build_arrival_shift_series, rolling_drift_scan,
                    detection_delay_days, false_alarm_count, DriftReport
  conformal       : three_way_time_split, conformal_quantile,
                    fit_and_calibrate, ConformalResult

See each module's docstring for design rationale.
"""
