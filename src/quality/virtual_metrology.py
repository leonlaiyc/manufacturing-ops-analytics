"""
Tabular virtual metrology (M7 Stage C).

Predicts ``metrology_reading`` (Stage B's noisy virtual-metrology stand-in,
see ``yield_model.py``) from STRICTLY UPSTREAM process variables using a
hand-built ordinary-least-squares regression (``numpy.linalg.lstsq``), no
``sklearn`` (project rule: interpretable, first-principles methods only).

Leakage safety
--------------
The feature set is built ONLY from information available before/at the METRO
step in a real fab (route order: CLEAN, FURNACE, DEPO, LITHO, ETCH, LITHO,
IMPLANT, METRO):

  - post-litho queue times for BOTH LITHO visits (gap1, gap2) and their
    violation flags (viol_litho1, viol_litho2)   -- from ``queue_time.py``
  - LITHO process durations for both visits (litho1_pt, litho2_pt)
  - ETCH process duration (etch_pt) and standardized excess (pt_excess_etch)
  - one-hot indicators for which physical tool ran each LITHO visit
    (litho1_is_toolB, litho2_is_toolB)

None of these touch the METRO step itself or any outcome/label column.
``p_latent``, ``defects``, and ``lot_yield`` are LABELS (the thing the
yield model derived FROM the same risk factors) and must never appear as
features here -- using them would leak the answer back into the predictor
of a noisy version of itself. ``metrology_reading`` is the regression
TARGET, also never a feature of itself.

Time-based split, not random
-----------------------------
Lots are sorted by their own completion time and split 70/30 (train = first
70%, test = last 30%). A random split would leak information across the
train/test boundary via shared tool state and queue dynamics: nearby lots in
time share the same LITHO tool occupancy, queue congestion, and (in the
chamber-matching scenario) the same tool-offset regime. Evaluating on a
held-out FUTURE window is the only split that matches how a real virtual-
metrology model would actually be deployed: fit on history, predict on lots
not yet metrology-inspected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from queue_time import post_litho_queue_times

#: Feature columns produced by ``build_features`` (fixed order; also the
#: order of the fitted coefficient table).
FEATURE_COLUMNS = [
    "gap1", "gap2",
    "viol_litho1", "viol_litho2",
    "litho1_pt", "litho2_pt",
    "etch_pt", "pt_excess_etch",
    "litho1_is_toolB", "litho2_is_toolB",
]

TRAIN_FRACTION = 0.70


def build_features(log: pd.DataFrame, route: list, lot_quality: pd.DataFrame,
                    window_hours: float) -> pd.DataFrame:
    """Assemble the leakage-safe feature table, one row per lot.

    Parameters
    ----------
    log : pd.DataFrame
        Stage-A event log (must carry ``tool_id``).
    route : list
        Route used to generate ``log`` (for locating LITHO/ETCH step_seq).
    lot_quality : pd.DataFrame
        Output of ``yield_model.build_lot_quality`` on the SAME log; used
        only for its violation flags, tool labels, pt_excess_etch, and the
        completion time needed for the time-based split -- NEVER for
        p_latent/defects/lot_yield (those are excluded by construction:
        this function never reads those columns).

    Returns
    -------
    pd.DataFrame indexed by lot_id with FEATURE_COLUMNS plus
    ``metrology_reading`` (target), ``defects`` (outcome, for AUC scoring
    only -- not a feature), and ``completion_time`` (split key, not a
    feature).
    """
    gaps = post_litho_queue_times(log, route)
    g1 = gaps[gaps["visit"] == 1][["lot_id", "gap"]].rename(columns={"gap": "gap1"})
    g2 = gaps[gaps["visit"] == 2][["lot_id", "gap"]].rename(columns={"gap": "gap2"})

    litho_steps = [i for i, s in enumerate(route) if s == "LITHO"]
    etch_step = route.index("ETCH")

    def _pt(step_seq: int, colname: str) -> pd.DataFrame:
        rows = log.loc[log["step_seq"] == step_seq,
                        ["lot_id", "process_start_time", "process_complete_time"]].copy()
        rows[colname] = rows["process_complete_time"] - rows["process_start_time"]
        return rows[["lot_id", colname]]

    litho1_pt = _pt(litho_steps[0], "litho1_pt")
    litho2_pt = _pt(litho_steps[1], "litho2_pt")
    etch_pt = _pt(etch_step, "etch_pt")

    # Completion time = the lot's last recorded process_complete_time in the
    # log (route-order-agnostic), used only to order lots for the time split.
    completion = (log.groupby("lot_id")["process_complete_time"].max()
                  .rename("completion_time").reset_index())

    feat = lot_quality[["lot_id", "viol_litho1", "viol_litho2",
                        "litho1_tool", "litho2_tool", "pt_excess_etch",
                        "metrology_reading", "defects"]].copy()
    feat = (feat.merge(g1, on="lot_id", how="left")
                .merge(g2, on="lot_id", how="left")
                .merge(litho1_pt, on="lot_id", how="left")
                .merge(litho2_pt, on="lot_id", how="left")
                .merge(etch_pt, on="lot_id", how="left")
                .merge(completion, on="lot_id", how="left"))

    # One-hot: "is the visit's tool the second/lexicographically-later tool
    # label" (e.g. LITHO-2 rather than LITHO-1). Generic to any tool_id
    # labeling; does not assume which tool is off-nominal (that is a Stage-B
    # concept, not used here to avoid coupling the VM feature set to the
    # chamber-matching experiment design).
    tool_labels = sorted(pd.concat([feat["litho1_tool"], feat["litho2_tool"]])
                        .dropna().unique())
    tool_b = tool_labels[-1] if len(tool_labels) > 1 else None
    feat["litho1_is_toolB"] = (feat["litho1_tool"] == tool_b).astype(float) if tool_b else 0.0
    feat["litho2_is_toolB"] = (feat["litho2_tool"] == tool_b).astype(float) if tool_b else 0.0

    feat["viol_litho1"] = feat["viol_litho1"].astype(float)
    feat["viol_litho2"] = feat["viol_litho2"].astype(float)

    for col in ["gap1", "gap2", "litho1_pt", "litho2_pt", "etch_pt"]:
        feat[col] = feat[col].fillna(0.0)

    feat = feat.dropna(subset=["completion_time"]).reset_index(drop=True)
    return feat.set_index("lot_id")


def time_split(feat: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by ``completion_time`` and split first ``train_fraction`` / rest.

    See module docstring "Time-based split" for why this is used instead of
    a random split.
    """
    ordered = feat.sort_values("completion_time", kind="mergesort")
    n_train = int(round(len(ordered) * train_fraction))
    return ordered.iloc[:n_train], ordered.iloc[n_train:]


@dataclass
class VMResult:
    """Fitted virtual-metrology model + evaluation artifacts."""
    coef_table: pd.DataFrame            # feature, coefficient, interpretation
    metrics: dict = field(default_factory=dict)
    y_test_pred: np.ndarray = None
    y_test_true: np.ndarray = None


_INTERPRETATIONS = {
    "intercept": "baseline predicted metrology reading with all features at 0",
    "gap1": "each extra hour of post-litho-visit-1 queue wait raises predicted reading",
    "gap2": "each extra hour of post-litho-visit-2 queue wait raises predicted reading",
    "viol_litho1": "breaching the visit-1 queue-time window raises predicted reading",
    "viol_litho2": "breaching the visit-2 queue-time window raises predicted reading",
    "litho1_pt": "each extra hour of LITHO visit-1 processing time shifts predicted reading",
    "litho2_pt": "each extra hour of LITHO visit-2 processing time shifts predicted reading",
    "etch_pt": "each extra hour of ETCH processing time shifts predicted reading",
    "pt_excess_etch": "each standardized unit of slow-side ETCH excess raises predicted reading",
    "litho1_is_toolB": "running LITHO visit 1 on the later-labeled tool shifts predicted reading",
    "litho2_is_toolB": "running LITHO visit 2 on the later-labeled tool shifts predicted reading",
}


def fit_ols(train: pd.DataFrame, feature_columns: list = FEATURE_COLUMNS
            ) -> tuple[np.ndarray, pd.DataFrame]:
    """Hand-built OLS via numpy.linalg.lstsq (design matrix + intercept column).

    Returns (coefficients incl. intercept as coef[0], coef_table DataFrame).
    """
    X = train[feature_columns].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    y = train["metrology_reading"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    names = ["intercept"] + feature_columns
    table = pd.DataFrame({
        "feature": names,
        "coefficient": coef,
        "interpretation": [_INTERPRETATIONS.get(n, "") for n in names],
    })
    return coef, table


def predict(coef: np.ndarray, df: pd.DataFrame,
            feature_columns: list = FEATURE_COLUMNS) -> np.ndarray:
    X = df[feature_columns].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ coef


def r2_vs_mean_baseline(y_true: np.ndarray, y_pred: np.ndarray,
                         train_mean: float) -> float:
    """Test R^2 against a TRAIN-MEAN baseline predictor (not test-mean).

    R^2 = 1 - SS_res / SS_tot, where SS_tot uses the constant baseline
    "always predict the train-set mean" (the honest no-model comparator: a
    model that only ever saw train-set y should be compared to a naive
    predictor that only ever saw train-set y too).
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - train_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def risk_ranking_auc(y_score: np.ndarray, y_outcome: np.ndarray) -> float:
    """AUC of ``y_score`` ranking against a binary ``y_outcome`` (defects>0).

    Hand-built via the rank-sum / Mann-Whitney U equivalence (no sklearn):
        AUC = (sum of ranks of positive-class scores - n_pos*(n_pos+1)/2)
              / (n_pos * n_neg)
    where ranks are computed over the pooled scores (average rank on ties).
    This is exactly the probability that a randomly chosen positive
    (defective) lot scores higher than a randomly chosen negative
    (defect-free) lot, i.e. the Mann-Whitney U statistic normalized to
    [0, 1].
    """
    y_outcome = np.asarray(y_outcome, dtype=bool)
    n_pos = int(y_outcome.sum())
    n_neg = int((~y_outcome).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    # Average-rank tie handling.
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed ranks
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = float(ranks[y_outcome].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def fit_and_evaluate(log: pd.DataFrame, route: list, lot_quality: pd.DataFrame,
                      window_hours: float,
                      feature_columns: list = FEATURE_COLUMNS,
                      train_fraction: float = TRAIN_FRACTION) -> VMResult:
    """End-to-end: build features, time-split, fit OLS, score test set.

    Metrics dict keys:
      test_r2        : test R^2 vs train-mean baseline predictor
      auc            : risk-ranking AUC of predicted reading vs defects>0 (test set)
      n_train, n_test: split sizes
    """
    feat = build_features(log, route, lot_quality, window_hours)
    train, test = time_split(feat, train_fraction)

    coef, coef_table = fit_ols(train, feature_columns)
    train_mean = float(train["metrology_reading"].mean())

    y_test_true = test["metrology_reading"].to_numpy(dtype=float)
    y_test_pred = predict(coef, test, feature_columns)

    test_r2 = r2_vs_mean_baseline(y_test_true, y_test_pred, train_mean)
    auc = risk_ranking_auc(y_test_pred, (test["defects"] > 0).to_numpy())

    metrics = {
        "test_r2": test_r2,
        "auc": auc,
        "n_train": len(train),
        "n_test": len(test),
        "train_mean_metrology_reading": train_mean,
    }
    return VMResult(coef_table=coef_table, metrics=metrics,
                    y_test_pred=y_test_pred, y_test_true=y_test_true)
