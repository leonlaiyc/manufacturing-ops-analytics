"""
M11 Stage A - leakage-safe as-of join.

Feature/label joins in this project (e.g. M7 virtual metrology, joining
upstream process features to a later metrology label) must only use feature
rows whose timestamp is STRICTLY EARLIER than the label's timestamp. Joining
on "less than or equal to" (the pandas ``merge_asof`` default,
``allow_exact_matches=True``) lets a feature row stamped at the EXACT same
instant as the label leak information that would not actually be available
yet at label time in a real deployment (ties are an artifact of the discrete
event log's timestamp resolution, not a physical guarantee of ordering).
Hence: ``direction="backward"`` (only look backward in time) combined with
``allow_exact_matches=False`` (strictly earlier, exact ties excluded).

``leakage_safe_asof_join`` wraps ``pandas.merge_asof`` with these two options
locked in. ``audit_join`` is a belt-and-braces post-hoc check: given any
already-joined frame, it flags rows where the feature timestamp is not
strictly earlier than the label timestamp, so a caller can verify a join
(including ones NOT built with this helper) is actually leakage-safe.
"""

from __future__ import annotations

import pandas as pd


def leakage_safe_asof_join(features_df: pd.DataFrame, labels_df: pd.DataFrame,
                            feature_time_col: str, label_time_col: str,
                            by: str | list) -> pd.DataFrame:
    """As-of join labels to the most recent STRICTLY EARLIER feature row.

    For each row in ``labels_df``, attaches the feature row (grouped by
    ``by``, e.g. lot_id) with the largest ``feature_time_col`` value that is
    strictly less than that label's ``label_time_col`` value. Feature rows
    at or after the label time are never eligible, closing the two ways a
    naive join can leak: allowing an exact-time match, or looking forward.

    Both inputs are sorted on their respective time columns internally
    (``merge_asof`` requires the "on" frame sorted); the caller's original
    frames are not mutated.
    """
    by_cols = [by] if isinstance(by, str) else list(by)

    left = labels_df.sort_values(label_time_col).reset_index(drop=True)
    right = features_df.sort_values(feature_time_col).reset_index(drop=True)

    joined = pd.merge_asof(
        left, right,
        left_on=label_time_col, right_on=feature_time_col,
        by=by_cols,
        direction="backward",
        allow_exact_matches=False,
    )
    return joined


def audit_join(joined: pd.DataFrame, feature_time_col: str, label_time_col: str) -> pd.DataFrame:
    """Belt-and-braces audit: rows where feature_time is NOT strictly earlier
    than label_time (i.e. missing feature_time is fine - no match found -
    but a present feature_time >= label_time is a leak).

    Returns the offending rows (empty DataFrame if the join is clean).
    """
    have_both = joined[feature_time_col].notna() & joined[label_time_col].notna()
    leaked = have_both & (joined[feature_time_col] >= joined[label_time_col])
    return joined.loc[leaked]
