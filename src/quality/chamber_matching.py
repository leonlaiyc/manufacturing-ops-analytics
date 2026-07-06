"""
Chamber matching: LITHO-1 vs LITHO-2 (M7 Stage C).

Compares the two LITHO tools on (a) their own processing durations and (b)
downstream metrology readings, using scipy's standard two-sample tests. This
is the classic fab "chamber matching" question: are nominally-identical
process chambers actually behaving the same, or has one drifted?

Only meaningful on a log generated WITH ``LITHO`` ``tool_offsets`` (e.g.
``(1.05, 0.95)``, see ``factory_generator.StationConfig``): with no offsets
the two tools are drawn from the identical distribution and the tests should
find nothing (verified by ``mismatch_verdict``'s false-positive guard, and by
GATE 3 of ``vm_check.py``).

Everything here is a statistical comparison of SYNTHETIC data; it does not
claim to detect a real chamber-matching issue, only to recover an injected
one under the project's fixed-seed, hand-built-methods rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardized mean difference (a - b)."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_sd)


def _rank_biserial(a: np.ndarray, b: np.ndarray, u_stat: float) -> float:
    """Rank-biserial effect size for Mann-Whitney U, in [-1, 1]."""
    n_a, n_b = len(a), len(b)
    return float(1.0 - 2.0 * u_stat / (n_a * n_b))


@dataclass
class TwoSampleResult:
    metric: str
    mean_tool1: float
    mean_tool2: float
    t_stat: float
    t_pvalue: float
    u_stat: float
    u_pvalue: float
    cohens_d: float
    rank_biserial: float


def compare_litho_tools(log: pd.DataFrame, route: list, lot_quality: pd.DataFrame,
                         tool1: str = "LITHO-1", tool2: str = "LITHO-2"
                         ) -> dict[str, TwoSampleResult]:
    """Two-sample t-test and Mann-Whitney U for durations and metrology reading.

    Parameters
    ----------
    log : pd.DataFrame
        Event log (must carry ``tool_id``), generated with LITHO tool_offsets
        for a meaningful (non-null) comparison.
    route : list
        Route used to generate ``log``.
    lot_quality : pd.DataFrame
        Output of ``yield_model.build_lot_quality`` on the SAME log, used
        for ``metrology_reading`` and the two ``lithoN_tool`` columns.
    tool1, tool2 : str
        The two tool_id labels to compare (default the standard 2-tool
        LITHO labels).

    Returns
    -------
    dict with keys "processing_duration" and "metrology_reading", each a
    ``TwoSampleResult``. Processing duration pools BOTH LITHO visits'
    process times, split by which physical tool served that visit (a lot
    contributes up to two duration observations, one per visit, each
    attributed to whichever tool ran it). Metrology reading is one
    observation per lot, split by whether tool1 or tool2 served EITHER of
    its LITHO visits (lots served by both are excluded from this
    comparison as ambiguous).
    """
    litho_steps = [i for i, s in enumerate(route) if s == "LITHO"]
    litho_rows = log.loc[log["step_seq"].isin(litho_steps),
                         ["lot_id", "tool_id", "process_start_time",
                          "process_complete_time"]].copy()
    litho_rows["duration"] = (litho_rows["process_complete_time"]
                              - litho_rows["process_start_time"])

    dur1 = litho_rows.loc[litho_rows["tool_id"] == tool1, "duration"].to_numpy()
    dur2 = litho_rows.loc[litho_rows["tool_id"] == tool2, "duration"].to_numpy()
    duration_result = _two_sample("processing_duration", dur1, dur2)

    served1 = (lot_quality["litho1_tool"] == tool1) | (lot_quality["litho2_tool"] == tool1)
    served2 = (lot_quality["litho1_tool"] == tool2) | (lot_quality["litho2_tool"] == tool2)
    exclusive1 = served1 & ~served2
    exclusive2 = served2 & ~served1
    read1 = lot_quality.loc[exclusive1, "metrology_reading"].to_numpy()
    read2 = lot_quality.loc[exclusive2, "metrology_reading"].to_numpy()
    reading_result = _two_sample("metrology_reading", read1, read2)

    return {"processing_duration": duration_result,
            "metrology_reading": reading_result}


def _two_sample(name: str, a: np.ndarray, b: np.ndarray) -> TwoSampleResult:
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return TwoSampleResult(
        metric=name,
        mean_tool1=float(a.mean()), mean_tool2=float(b.mean()),
        t_stat=float(t_stat), t_pvalue=float(t_p),
        u_stat=float(u_stat), u_pvalue=float(u_p),
        cohens_d=_cohens_d(a, b),
        rank_biserial=_rank_biserial(a, b, float(u_stat)),
    )


def daily_mean_series(log: pd.DataFrame, route: list,
                       tool1: str = "LITHO-1", tool2: str = "LITHO-2"
                       ) -> pd.DataFrame:
    """Per-chamber daily mean LITHO processing duration (for a control-chart figure).

    Returns a tidy DataFrame: day, tool_id, mean_duration, n -- one row per
    (day, tool) with at least one LITHO operation. ``day = floor(process_
    complete_time / 24)``, matching the daily-bucketing convention used
    elsewhere in this repo (see ``cost_model.daily_operating_cost``).
    """
    litho_steps = [i for i, s in enumerate(route) if s == "LITHO"]
    rows = log.loc[log["step_seq"].isin(litho_steps) & log["tool_id"].isin([tool1, tool2]),
                   ["tool_id", "process_start_time", "process_complete_time"]].copy()
    rows["duration"] = rows["process_complete_time"] - rows["process_start_time"]
    rows["day"] = np.floor(rows["process_complete_time"] / 24.0).astype(int)
    out = (rows.groupby(["day", "tool_id"])["duration"]
           .agg(mean_duration="mean", n="count")
           .reset_index())
    return out.sort_values(["day", "tool_id"]).reset_index(drop=True)


def mismatch_verdict(results: dict[str, TwoSampleResult], alpha: float = 0.01
                      ) -> dict[str, str]:
    """"detected" / "not detected" per metric at significance level ``alpha``.

    Uses the Mann-Whitney p-value (distribution-free, robust to the skew of
    lognormal processing times) as the primary test; a metric is "detected"
    when ``u_pvalue < alpha``. Designed to be silent (all "not detected") on
    an offset-free log, where the two tools are drawn from the identical
    distribution -- verified as a false-positive guard by
    ``vm_check.py`` GATE 3.
    """
    return {name: ("detected" if r.u_pvalue < alpha else "not detected")
            for name, r in results.items()}
