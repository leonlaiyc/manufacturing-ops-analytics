"""
M11 Stage B - rolling-window drift monitor.

A hand-built, first-principles drift detector: no external drift-detection
library, same "explainable from first principles" standard as the M5
control-chart / EWMA detectors (``monitoring/detectors.py``), applied here to
DATA-QUALITY-style channels (a station's daily mean process time, and the
daily arrival count) rather than the KPI series M5 already watches.

Method
------
1. Fix a REFERENCE window on early, known-clean data: ``[ref_start, ref_end)``
   days. Compute its mean and standard deviation once, up front. This mirrors
   ``detectors.fit_baseline`` (leakage-free: only clean, pre-drift days feed
   the reference).
2. Slide a TEST window of length ``test_window_days`` one day at a time over
   the rest of the series. For each test-window position ending on day t,
   compute its mean over ``[t - test_window_days + 1, t]``.
3. Score each test-window position as a STANDARDIZED DIFFERENCE OF MEANS
   (a two-sample z-style statistic, not a t-test p-value, since only a
   number-of-standard-errors distance is needed, not a formal hypothesis
   test):

       score(t) = (test_mean(t) - ref_mean) / se

   where ``se = ref_std / sqrt(ref_window_days)`` is the standard error of
   the REFERENCE mean (the test window is compared against the reference's
   own precision, so a longer/more-precise reference window yields a more
   sensitive detector; the test window's own internal variance is not part
   of ``se`` by design, since the question is "has the level moved away from
   the reference", not "is the test window internally noisy").
4. ALARM when ``|score(t)| >= z_threshold`` for ``k_consecutive`` consecutive
   test-window positions in a row (a persistence requirement, so one noisy
   day cannot trip the alarm alone; this is the same "require sustained
   evidence" idea as EWMA's smoothing, done here as a simple run-length rule
   instead of exponential weighting, since the drift monitor is meant to be
   the simplest possible baseline, not a second EWMA).

Defaults: ``test_window_days = 5``, ``z_threshold = 7.0``, ``k_consecutive =
3``. The threshold is much higher than the M5 control chart's k=3 sigma
because this monitor's score is already a STANDARD ERROR of a WINDOW MEAN
(much less noisy than a single raw day) evaluated on a station running near
its bottleneck utilization (LITHO, slot rho ~= 0.85): near-saturation queues
have natural autocorrelated wobble that produces sustained multi-day runs
away from the reference mean even with NO injected drift at all. A
control-chart-scale threshold (z~3-4) false-alarms repeatedly on this
natural wobble; z=7.0 combined with a 3-in-a-row persistence rule was swept
empirically (see ``reliability_check.py`` GATE 1/2) to be the smallest
threshold in the swept grid that gives zero alarms on a clean run while
still catching the injected drift with a finite, reported delay.

Channels monitored (at least two, per the M11 Stage B spec):
  - ``daily_mean_process_time``: a station's daily mean RAW process time
    (process_complete_time - process_start_time), e.g. LITHO. Rises under a
    ``DegradationAnomaly`` (slow processing-time ramp).
  - ``daily_arrival_count``: lots arriving per day. Shifts under a change in
    the arrival-rate regime (see ``build_arrival_shift_series`` below).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 24.0

TEST_WINDOW_DAYS = 5
Z_THRESHOLD = 7.0
K_CONSECUTIVE = 3


def daily_mean_process_time(log: pd.DataFrame, station: str) -> pd.Series:
    """Daily mean RAW process time (complete - start) at ``station``.

    Ops are bucketed by the day their processing completes (same convention
    as ``monitoring.kpi_series.daily_bottleneck_wait``). For a re-entrant
    station (e.g. LITHO) both visits per lot are included.
    """
    ops = log.loc[log["station"] == station].copy()
    ops["pt"] = ops["process_complete_time"] - ops["process_start_time"]
    ops["day"] = np.floor(ops["process_complete_time"] / DAY).astype(int)
    s = ops.groupby("day")["pt"].mean()
    s.name = "daily_mean_process_time"
    return s


def daily_arrival_count(lifecycle: pd.DataFrame, horizon_hours: float) -> pd.Series:
    """Lots arriving per day, dense over ``[0, horizon_hours/24)`` (0 for empty days)."""
    df = lifecycle.copy()
    df["day"] = np.floor(df["arrival_time"] / DAY).astype(int)
    counts = df.groupby("day").size()
    days = range(int(np.ceil(horizon_hours / DAY)))
    s = counts.reindex(days, fill_value=0)
    s.name = "daily_arrival_count"
    return s


def build_arrival_shift_series(lifecycle_before: pd.DataFrame, horizon_before: float,
                                lifecycle_after: pd.DataFrame, horizon_after: float
                               ) -> pd.Series:
    """Stylized arrival-rate cut-over: concatenate two runs' daily arrival counts.

    There is no dedicated "arrival-rate step change" injection primitive in
    the generator (only ``DemandSurgeAnomaly``, a temporary window). To
    exercise the drift monitor on a SUSTAINED rate shift, two independent
    simulation runs at different ``cfg.arrival_rate`` are concatenated day
    after day: days ``[0, horizon_before/24)`` from the lower-rate run, then
    days ``[horizon_before/24, horizon_before/24 + horizon_after/24)`` from
    the higher-rate run, re-indexed to be contiguous. This is explicitly
    stylized (no real fab data), documented here and at every call site, and
    used ONLY to exercise the drift monitor's arrival-count channel end to
    end, never presented as a forecast of a real cut-over.
    """
    before = daily_arrival_count(lifecycle_before, horizon_before)
    after = daily_arrival_count(lifecycle_after, horizon_after)
    n_before = len(before)
    after = after.reset_index(drop=True)
    after.index = after.index + n_before
    combined = pd.concat([before.reset_index(drop=True).set_axis(range(n_before)), after])
    combined.name = "daily_arrival_count"
    combined.index.name = "day"
    return combined


@dataclass
class DriftReport:
    """Result of one rolling-window drift scan over a series."""
    channel: str
    ref_start: int
    ref_end: int
    ref_mean: float
    ref_std: float
    test_window_days: int
    z_threshold: float
    k_consecutive: int
    scores: pd.Series          # index = day the test window ends on
    alarms: pd.Series          # bool, same index as scores
    first_alarm_day: int | None  # first day the k-consecutive rule fires


def rolling_drift_scan(series: pd.Series, ref_start: int, ref_end: int,
                        test_window_days: int = TEST_WINDOW_DAYS,
                        z_threshold: float = Z_THRESHOLD,
                        k_consecutive: int = K_CONSECUTIVE,
                        channel: str = "") -> DriftReport:
    """Scan ``series`` for drift relative to a fixed reference window.

    Parameters
    ----------
    series : pd.Series
        Daily channel values, indexed by day (int), e.g. from
        ``daily_mean_process_time`` or ``daily_arrival_count``.
    ref_start, ref_end : int
        Reference window ``[ref_start, ref_end)`` days (must be clean, known
        pre-drift data; leakage-free by construction since the caller
        chooses this window from data known in advance to predate any
        injected drift).
    test_window_days : int
        Length of the sliding test window (days).
    z_threshold, k_consecutive : see module docstring.

    Returns
    -------
    DriftReport. ``first_alarm_day`` is the day index the test window ENDS
    on for the first position where ``k_consecutive`` consecutive scores
    all exceed the threshold in magnitude (None if never).
    """
    ref_vals = series.reindex(range(ref_start, ref_end)).dropna()
    ref_mean = float(ref_vals.mean())
    ref_std = float(ref_vals.std(ddof=1))
    se = ref_std / np.sqrt(len(ref_vals))

    last_day = int(series.index.max())
    score_days = list(range(ref_end + test_window_days - 1, last_day + 1))
    scores = []
    for t in score_days:
        window = series.reindex(range(t - test_window_days + 1, t + 1)).dropna()
        test_mean = float(window.mean()) if len(window) else float("nan")
        score = (test_mean - ref_mean) / se if se > 0 else float("nan")
        scores.append(score)
    scores = pd.Series(scores, index=score_days, name="score")

    exceeds = (scores.abs() >= z_threshold).fillna(False)
    alarms = pd.Series(False, index=scores.index)
    run = 0
    first_alarm_day = None
    for day, hit in exceeds.items():
        run = run + 1 if hit else 0
        if run >= k_consecutive:
            alarms.loc[day] = True
            if first_alarm_day is None:
                first_alarm_day = day

    return DriftReport(
        channel=channel, ref_start=ref_start, ref_end=ref_end,
        ref_mean=ref_mean, ref_std=ref_std,
        test_window_days=test_window_days, z_threshold=z_threshold,
        k_consecutive=k_consecutive, scores=scores, alarms=alarms,
        first_alarm_day=first_alarm_day,
    )


def detection_delay_days(report: DriftReport, onset_day: int) -> int | None:
    """Days between a known injection ``onset_day`` and the first alarm.

    None if the monitor never alarmed. Negative values are possible in
    principle (alarm before onset) but would indicate a false alarm bleeding
    into the delay calculation, not genuine early detection; callers should
    treat a negative delay as a red flag, not a good result.
    """
    if report.first_alarm_day is None:
        return None
    return int(report.first_alarm_day - onset_day)


def false_alarm_count(report: DriftReport, exclude_before: int = 0) -> int:
    """Count alarm days at/after ``exclude_before`` (e.g. to skip a known
    injection window when scoring a clean-run baseline over the full series).
    """
    return int(report.alarms.loc[report.alarms.index >= exclude_before].sum())
