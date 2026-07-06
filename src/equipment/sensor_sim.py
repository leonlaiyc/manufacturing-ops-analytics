"""
Simulated multivariate tool-sensor signatures (M8 Stage C).

FRAMING (read this before using any result from this module or from
``pdm_model.py``): every channel generated here is a SELF-GENERATED synthetic
signal. The generative rule that couples two of the channels to a degradation
ramp is written by this project, with a fixed seed, on purpose. Any model
later trained on these channels can therefore only ever RE-DISCOVER the rule
this module already encodes - it cannot demonstrate predictive power on a
real fab, because there is no real fab underneath it. Every use of this data
in this repo is scored as MEASURING DETECTION QUALITY AGAINST KNOWN GROUND
TRUTH (do the two wired channels get recovered as top drivers? does a health
score raised from them separate degraded from clean tool-days on the labels
we ourselves attached?), never as evidence of real predictive maintenance
(PdM) capability. See CLAUDE.md's owner-approved M8 exception.

Why a standalone daily generator, not per-event DES sensor draws
------------------------------------------------------------------
Real fab tool sensors report at a fixed cadence (temperature, vibration,
pressure, flow, current) independent of how many lots the tool happened to
process that day. This module mirrors that: one row per (tool, day), not per
DES event. It does not read the event log or run the DES at all - it is a
self-contained generative layer whose only link to the rest of this project
is sharing the vocabulary of ``factory_generator.DegradationAnomaly``
(severity ramps as ``alpha * (t - t_onset)``) so the coupling is recognizable
to a reader of this codebase, not because it depends on the DES output.

Generative model (per tool, per channel, per day t)
------------------------------------------------------
    channel(t) = nominal_level
               + ar1_noise(t)                      (all channels)
               + degradation_term(t)                (two wired channels only)

  ar1_noise: a stationary AR(1) process,
      noise(t) = phi * noise(t-1) + eps(t),  eps(t) ~ Normal(0, innovation_sd)
  so consecutive days are correlated (a real sensor does not reset to
  independent noise every day) but the process is mean-reverting (no unit
  root, no drift of its own - any trend seen in a wired channel comes only
  from the degradation term below).

  degradation_term: exactly TWO of the channels (documented per station in
  ``DEGRADATION_WIRING``) are coupled to the injected severity ramp. Reusing
  ``factory_generator.DegradationAnomaly``'s own multiplier shape (linear
  ramp from onset) AND its reset-to-normal semantics at ``t_end``, each wired
  channel k applies
      degradation_term_k(t) = gain_k * severity(t - lag_k)
      severity(u) = 0                          if u < 0 (before onset)
                  = alpha * u                  if 0 <= u < duration
                  = 0                           if u >= duration (repaired:
                                                 mirrors DegradationAnomaly
                                                 resetting its multiplier to
                                                 1.0 once t passes t_end)
  with two DIFFERENT gains (one channel reacts more strongly) and one channel
  carrying a small LAG (its degradation term uses severity at (t - lag), i.e.
  it shows the ramp a few days later than the other wired channel) - this is
  the deliberate multivariate structure GATE 4 (in ``pdm_check.py``) checks
  SHAP recovers: two true drivers, of different strength and timing, buried
  among pure-noise distractor channels. The episode horizon extends well past
  ``onset + duration`` so the evaluation's "clean again after recovery" days
  (see ``monitoring.evaluation``'s grace-period convention) are genuinely
  free of the degradation term, matching M5's own false-alarm-rate discipline.

  All other channels are pure noise/distractors: same AR(1) noise model, no
  degradation_term at all. They exist so the health model and SHAP have
  something to correctly IGNORE, not just channels to correctly use.

Episode design
--------------
An "episode" is one independent simulated run of one station's tools over a
fixed horizon, with its own numpy ``Generator`` seed, its own randomly-drawn
degradation onset day and severity (``alpha``), on ONE tool designated the
"degrading tool" for that episode (chosen by index, fixed per episode so it
is reproducible). Every other tool in the same episode runs clean (noise
only) for the whole horizon, so each episode contains both degraded and
clean tool-days for the classifier and for the EWMA comparison baseline.
Ground truth per (tool, day): ``degraded`` (bool) and ``days_since_onset``
(NaN when not degraded or before onset), attached directly to the returned
frame - never inferred, always the exact values used to generate the row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Sensor channels simulated per tool per day. Five channels, matching the
#: "temperature, vibration, pressure, flow, current" analog list in the spec.
CHANNELS = ("temperature", "vibration", "pressure", "flow", "current")

#: Per-channel (nominal_level, ar1_phi, innovation_sd), chosen so channels sit
#: on plausible, well-separated scales (illustrative, not calibrated to any
#: real tool datasheet - this is a synthetic layer, see module docstring).
CHANNEL_BASELINE = {
    "temperature": {"nominal_level": 450.0, "phi": 0.6, "innovation_sd": 1.5},
    "vibration": {"nominal_level": 2.0, "phi": 0.5, "innovation_sd": 0.08},
    "pressure": {"nominal_level": 100.0, "phi": 0.7, "innovation_sd": 0.6},
    "flow": {"nominal_level": 50.0, "phi": 0.6, "innovation_sd": 0.5},
    "current": {"nominal_level": 12.0, "phi": 0.5, "innovation_sd": 0.15},
}

#: Which two channels are wired to the degradation ramp, per station, and
#: with what (gain, lag_days). "vibration" reacts fastest and strongest (a
#: tool running degraded vibrates more, no lag); "temperature" reacts more
#: slowly and with a 3-day lag (heat buildup takes time to show at the
#: sensor). Every other channel in CHANNELS is a pure-noise distractor for
#: that station. Documented here once; ``simulate_station_episode`` reads it.
DEGRADATION_WIRING = {
    "LITHO": {
        "vibration": {"gain": 0.35, "lag_days": 0},
        "temperature": {"gain": 6.0, "lag_days": 3},
    },
}

#: Documented range degradation severity (alpha, per-day fractional
#: inflation) and onset day are drawn from for each simulated episode. Kept
#: modest so the ramp stays in a "slow drift" regime for the whole episode
#: horizon, consistent with DegradationAnomaly's own framing.
ALPHA_RANGE = (0.02, 0.06)
ONSET_DAY_RANGE = (15, 40)


@dataclass
class EpisodeConfig:
    """One episode's simulation parameters (drawn or fixed, always recorded)."""
    episode_id: int
    seed: int
    station: str
    n_tools: int
    horizon_days: int
    degrading_tool_index: int
    onset_day: int
    alpha: float
    duration_days: int


def _ar1_noise(rng: np.random.Generator, n_days: int, phi: float,
               innovation_sd: float) -> np.ndarray:
    """Stationary AR(1) noise path of length ``n_days``, mean 0.

    ``noise(t) = phi * noise(t-1) + eps(t)``, started from the process's own
    stationary distribution (``Normal(0, innovation_sd / sqrt(1 - phi**2))``)
    so the first simulated day is not artificially at 0 (no burn-in bias).
    """
    eps = rng.normal(0.0, innovation_sd, size=n_days)
    noise = np.empty(n_days)
    noise[0] = rng.normal(0.0, innovation_sd / np.sqrt(1.0 - phi ** 2))
    for t in range(1, n_days):
        noise[t] = phi * noise[t - 1] + eps[t]
    return noise


def _severity_path(n_days: int, onset_day: int, alpha: float,
                   duration_days: int, lag_days: int = 0) -> np.ndarray:
    """Severity(t - lag_days) per day: 0 before onset, linear ramp, 0 after repair.

    Mirrors ``factory_generator.DegradationAnomaly``'s linear-ramp shape
    (``alpha * (t - t_onset)``) AND its reset-to-normal behaviour once past
    ``t_end``: severity is 0 for elapsed < 0 (before onset), ramps linearly
    for ``0 <= elapsed < duration_days``, then drops back to 0 (repaired) for
    ``elapsed >= duration_days``. The episode's ground-truth ``degraded``
    label (see ``simulate_station_episode``) uses this same
    ``[onset, onset + duration_days)`` window, so "recovered" days are
    genuinely free of the degradation term for the false-alarm evaluation.
    """
    days = np.arange(n_days) - lag_days
    elapsed = days - onset_day
    sev = np.where((elapsed >= 0) & (elapsed < duration_days), elapsed * alpha, 0.0)
    return sev


def simulate_station_episode(cfg: EpisodeConfig) -> pd.DataFrame:
    """Simulate one episode: all tools of ``cfg.station``, ``cfg.horizon_days`` days.

    Returns a tidy per-(tool, day) frame with columns:
        episode_id, tool_id, day, <CHANNELS...>, degraded, days_since_onset
    ``degraded`` and ``days_since_onset`` are the ground truth attached at
    generation time (never inferred): True / non-NaN only for the episode's
    designated degrading tool, on days at or after ``onset_day``.
    """
    rng = np.random.default_rng(cfg.seed)
    wiring = DEGRADATION_WIRING[cfg.station]
    rows = []
    for tool_idx in range(cfg.n_tools):
        tool_id = f"{cfg.station}-{tool_idx + 1}"
        is_degrading_tool = (tool_idx == cfg.degrading_tool_index)

        channel_paths = {}
        for ch in CHANNELS:
            base = CHANNEL_BASELINE[ch]
            noise = _ar1_noise(rng, cfg.horizon_days, base["phi"], base["innovation_sd"])
            level = base["nominal_level"] + noise
            if is_degrading_tool and ch in wiring:
                w = wiring[ch]
                sev = _severity_path(cfg.horizon_days, cfg.onset_day, cfg.alpha,
                                     cfg.duration_days, lag_days=w["lag_days"])
                level = level + w["gain"] * sev
            channel_paths[ch] = level

        days = np.arange(cfg.horizon_days)
        if is_degrading_tool:
            degraded = (days >= cfg.onset_day) & (days < cfg.onset_day + cfg.duration_days)
            days_since_onset = np.where(degraded, days - cfg.onset_day, np.nan)
        else:
            degraded = np.zeros(cfg.horizon_days, dtype=bool)
            days_since_onset = np.full(cfg.horizon_days, np.nan)

        tool_df = pd.DataFrame({
            "episode_id": cfg.episode_id,
            "tool_id": tool_id,
            "day": days,
            **channel_paths,
            "degraded": degraded,
            "days_since_onset": days_since_onset,
        })
        rows.append(tool_df)

    return pd.concat(rows, ignore_index=True)


def make_episodes(n_episodes: int, station: str = "LITHO", n_tools: int = 2,
                  horizon_days: int = 60, duration_days: int = 30,
                  base_seed: int = 9000,
                  alpha_range: tuple = ALPHA_RANGE,
                  onset_day_range: tuple = ONSET_DAY_RANGE) -> list:
    """Build ``n_episodes`` independent episodes with randomized onset/severity.

    Each episode gets its own seed (``base_seed + episode_id``) so episodes
    are independent and reproducible; the onset day, severity ``alpha``, and
    which tool degrades are all drawn from that episode's own
    ``np.random.default_rng(base_seed + episode_id)`` BEFORE simulating (a
    separate draw stream from the noise generation inside
    ``simulate_station_episode``, so changing ``horizon_days`` or ``n_tools``
    does not perturb which onset/alpha/tool an episode drew).

    Returns a list of dicts: {"config": EpisodeConfig, "frame": pd.DataFrame}.
    """
    if station not in DEGRADATION_WIRING:
        raise ValueError(f"no DEGRADATION_WIRING documented for station {station!r}")

    episodes = []
    for episode_id in range(n_episodes):
        draw_rng = np.random.default_rng(base_seed + episode_id)
        degrading_tool_index = int(draw_rng.integers(0, n_tools))
        onset_day = int(draw_rng.integers(onset_day_range[0], onset_day_range[1] + 1))
        alpha = float(draw_rng.uniform(alpha_range[0], alpha_range[1]))

        ecfg = EpisodeConfig(
            episode_id=episode_id,
            seed=base_seed + episode_id,
            station=station,
            n_tools=n_tools,
            horizon_days=horizon_days,
            degrading_tool_index=degrading_tool_index,
            onset_day=onset_day,
            alpha=alpha,
            duration_days=duration_days,
        )
        frame = simulate_station_episode(ecfg)
        episodes.append({"config": ecfg, "frame": frame})
    return episodes
