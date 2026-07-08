"""
M11 Stage A - event-log schema contract, corruption injectors, and a
leakage-safe as-of join helper.

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

Import convention (matches the rest of ``src/``): modules use BARE imports
(``from schema_contract import ...``) and consumers put ``src/dataquality``
on ``sys.path`` -- see ``dq_check.py`` for the pattern.

Public exports by module:

  schema_contract : REQUIRED_COLUMNS, NUMERIC_TIME_COLUMNS,
                    IDENTIFIER_COLUMNS, LOCKED_ROUTE, CLAUSES,
                    CLAUSE_DESCRIPTIONS, validate_log,
                    check_c1_columns_and_dtypes .. check_c6_tool_consistency
  corruptions     : INJECTORS, stringify_timestamps, drop_timestamps,
                    negate_duration, duplicate_rows, impossible_hop,
                    orphan_tool, shuffle_step_seq
  asof_join       : leakage_safe_asof_join, audit_join

See each module's docstring for design rationale.
"""
