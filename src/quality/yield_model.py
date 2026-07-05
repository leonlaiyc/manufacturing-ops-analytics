"""
Lot-level yield ground truth (M7 Stage B).

Turns the Stage-A event log (LITHO queue-time violations from
``queue_time.py``, plus each lot's own ETCH processing time and which LITHO
tool served it) into a per-lot LATENT defect probability, a REALIZED defect
count, a resulting lot yield, and a noisy "metrology reading" that stands in
for a virtual-metrology sensor (Stage C's prediction target).

Everything here is synthetic and stylized — a hand-built additive risk score,
not a physical yield model and not a claim about real-fab yield behavior. See
``CLAUDE.md`` honest-scope rule.

Why linear-additive, not logistic
------------------------------------
The latent defect probability is built as a plain SUM of independent additive
terms, clipped to ``[0, 0.95]``:

    p = clip(p_base + a_viol1*1{viol1} + a_viol2*1{viol2}
             + a_chamber*1{off-nominal tool} + a_pt_excess*pt_excess_std,  0, 0.95)

A logistic/log-odds link (``p = sigmoid(beta0 + beta1*x1 + ...)``) would be
the "standard" choice for a probability model, but it trades away exactly the
property this project's interpretability rule (CLAUDE.md: "if a technique
cannot be explained from first principles, choose a simpler one") asks for:
under a logit link, each coefficient's effect on p is a MULTIPLICATIVE, curve-
dependent statement about odds, so "a_viol1 = 0.06" cannot be read off as "a
queue-time violation adds about 6 percentage points of risk" without also
knowing where on the curve you are. Under the linear-additive form here, that
IS the reading, exactly, everywhere in the interior of the clip range — which
is the point: every coefficient is a literal, constant, first-principles
"probability points added by this condition," directly settable and
auditable by the project owner. The clip to ``[0, 0.95]`` is the one place
the linearity is deliberately broken, and only at the extremes, to keep p a
valid probability.

Realized vs. latent
-----------------------
``p`` is the LATENT (unobservable in a real fab) per-wafer defect
probability. The REALIZED outcome for a 25-wafer lot is one Binomial(25, p)
draw (independent across lots) from a dedicated, seeded ``numpy.random.
Generator`` — kept separate from the DES's own RNG/CRN draws so the yield
layer never perturbs Stage A/M2-M5 reproducibility. ``lot_yield = (25 -
defects) / 25``.

Metrology reading
--------------------
``metrology_reading = p + Normal(0, sigma)`` (same seeded generator, no clip):
a noisy observation of the latent risk, standing in for a real virtual-
metrology sensor. This is the Stage C prediction TARGET, not a feature; Stage
C aims to predict something close to ``metrology_reading`` (or ``p``) from
upstream process variables without seeing ``p`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from queue_time import flag_violations, DEFAULT_WINDOW_HOURS

#: Wafers per lot (locked design decision; see CLAUDE.md).
WAFERS_PER_LOT = 25

#: ETCH is a single, non-re-entrant visit in the locked route (step_seq 4).
ETCH_STATION = "ETCH"


@dataclass
class QualityConfig:
    """Parameters of the additive latent-defect-probability model.

    Attributes
    ----------
    p_base : float
        Baseline per-wafer defect probability with no risk factors present.
    a_viol1, a_viol2 : float
        Additive probability added when the lot's first / second LITHO visit
        breaches the post-litho queue-time window (``queue_time.py``).
        Modeled separately (not one shared coefficient) because a real fab
        would expect risk to compound across mask layers, and keeping them
        separate lets GATE 3 test monotonicity per visit.
    a_chamber : float
        Additive probability added when the lot's LITHO visit ran on the
        OFF-NOMINAL tool, i.e. the tool with the slower ``tool_offsets``
        multiplier (see ``factory_generator.StationConfig.tool_offsets``).
        Only meaningful when the log was generated WITH LITHO tool_offsets
        (e.g. ``(1.05, 0.95)``); on a log with no offsets there is no
        "off-nominal" tool and this effect must be disabled. The caller
        supplies which tool label is off-nominal via
        ``off_nominal_tool_label`` (below) — this module does not infer it
        from the offsets tuple, since a log carries only tool_id strings,
        not the offsets that produced them.
    a_pt_excess : float
        Additive probability per standardized unit of POSITIVE ETCH
        process-time excess (see ``etch_pt_excess`` below), i.e. running
        slower than the ETCH station's nominal mean. Excess is standardized
        by the station's nominal (pt_mean, pt_cv) and clipped at +2 standard
        units so one extreme outlier lot cannot dominate the sum.
    off_nominal_tool_label : str | None
        Tool_id string (e.g. ``"LITHO-2"``) that is the OFF-NOMINAL (slower)
        LITHO tool for the log being scored. ``None`` (default) disables the
        chamber effect entirely, regardless of ``a_chamber``, since with no
        label there is nothing to compare tool_id against.
    seed : int
        Seeds the DEDICATED ``numpy.random.Generator`` used for the Binomial
        defect draw and the metrology noise. Independent of the DES's own
        seed/CRN machinery (queue_time.py and the event log are read-only
        inputs here) so re-scoring a log never touches simulation
        reproducibility.
    """
    p_base: float = 0.02
    a_viol1: float = 0.06
    a_viol2: float = 0.08
    a_chamber: float = 0.03
    a_pt_excess: float = 0.04
    off_nominal_tool_label: str | None = None
    seed: int = 42


#: Standard deviation of the metrology-reading Gaussian noise (hours-free;
#: same units as the latent probability p, i.e. a probability-scale reading).
DEFAULT_METROLOGY_SIGMA = 0.01

#: Clip bound applied to standardized positive ETCH process-time excess.
PT_EXCESS_CLIP = 2.0


def etch_pt_excess(log: pd.DataFrame, route: list,
                    pt_mean: float, pt_cv: float) -> pd.DataFrame:
    """Per-lot standardized POSITIVE ETCH process-time excess.

    ETCH is a single (non-re-entrant) visit at route position
    ``route.index("ETCH")`` (step_seq 4 in the locked route). For each lot:

        raw_excess  = actual_process_time - pt_mean
        std_excess  = raw_excess / (pt_mean * pt_cv)      # lognormal's own SD
        pt_excess   = clip(std_excess, 0, PT_EXCESS_CLIP)  # only slow-side risk

    ``pt_mean * pt_cv`` is the ETCH station's own theoretical standard
    deviation (a lognormal with target mean ``pt_mean`` and coefficient of
    variation ``pt_cv`` has SD exactly ``pt_mean * pt_cv``; see
    ``factory_generator._lognormal_params``), so this standardizes against the
    station's OWN nominal variability rather than an arbitrary scale. Negative
    (faster-than-nominal) excess is clipped to 0 — running fast is not treated
    as a risk factor, only running slow is (an ETCH run that lingers is the
    stylized risk story here, e.g. over-etch).

    Returns
    -------
    pd.DataFrame: lot_id, etch_process_time, pt_excess_etch
    """
    etch_step = route.index(ETCH_STATION)
    etch_rows = log.loc[log["step_seq"] == etch_step,
                         ["lot_id", "process_start_time", "process_complete_time"]].copy()
    etch_rows["etch_process_time"] = (etch_rows["process_complete_time"]
                                       - etch_rows["process_start_time"])
    raw_excess = etch_rows["etch_process_time"] - pt_mean
    std_excess = raw_excess / (pt_mean * pt_cv)
    etch_rows["pt_excess_etch"] = std_excess.clip(lower=0.0, upper=PT_EXCESS_CLIP)
    return etch_rows[["lot_id", "etch_process_time", "pt_excess_etch"]]


def build_lot_quality(log: pd.DataFrame, route: list, cfg: QualityConfig,
                       window_hours: float = DEFAULT_WINDOW_HOURS,
                       etch_pt_mean: float = 1.0, etch_pt_cv: float = 0.5,
                       metrology_sigma: float = DEFAULT_METROLOGY_SIGMA
                       ) -> pd.DataFrame:
    """Build the tidy per-lot yield ground-truth table.

    Parameters
    ----------
    log : pd.DataFrame
        Stage-A event log (must carry ``tool_id``).
    route : list
        The route used to generate ``log`` (``cfg.route`` from the DES
        config; NOT this module's ``QualityConfig``).
    cfg : QualityConfig
        Yield-model parameters (see class docstring).
    window_hours : float
        Post-litho queue-time window passed through to
        ``queue_time.flag_violations``; defaults to the fixed, calibrated
        ``queue_time.DEFAULT_WINDOW_HOURS``.
    etch_pt_mean, etch_pt_cv : float
        ETCH station's nominal (pt_mean, pt_cv), used to standardize process-
        time excess (see ``etch_pt_excess``). Defaults match
        ``factory_generator.default_config``'s ETCH station; pass the actual
        config's values if scoring a log built with a different ETCH setup.
    metrology_sigma : float
        Standard deviation of the metrology-reading noise term.

    Returns
    -------
    pd.DataFrame, one row per lot, columns:
        lot_id             : lot identifier
        viol_litho1        : bool, first LITHO visit breached the queue-time
                              window (see queue_time.py; False if the lot has
                              no recorded first-visit-to-next-step gap)
        viol_litho2        : bool, second LITHO visit breached the window
        litho1_tool        : tool_id that ran the first LITHO visit (or NaN)
        litho2_tool        : tool_id that ran the second LITHO visit (or NaN)
        pt_excess_etch     : standardized positive ETCH process-time excess
                              (0 if the lot has no recorded ETCH row)
        p_latent           : latent per-wafer defect probability (see module
                              docstring for the additive formula)
        defects            : realized defective-wafer count, Binomial(25, p_latent)
        lot_yield          : (25 - defects) / 25
        metrology_reading  : p_latent + Gaussian noise (virtual-metrology
                              stand-in target; NOT a feature for Stage C)

    Every lot present in ``log`` appears exactly once, even if it has no
    recorded LITHO or ETCH row (incomplete lot at the end of the horizon);
    missing violation flags default to False and missing pt_excess defaults
    to 0, i.e. "no evidence of risk" rather than a fabricated risk score.
    """
    lots = pd.DataFrame({"lot_id": sorted(log["lot_id"].unique())})

    viol = flag_violations(log, route, window_hours=window_hours)
    v1 = (viol[viol["visit"] == 1][["lot_id", "violation", "litho_tool"]]
          .rename(columns={"violation": "viol_litho1", "litho_tool": "litho1_tool"}))
    v2 = (viol[viol["visit"] == 2][["lot_id", "violation", "litho_tool"]]
          .rename(columns={"violation": "viol_litho2", "litho_tool": "litho2_tool"}))

    out = lots.merge(v1, on="lot_id", how="left").merge(v2, on="lot_id", how="left")
    out["viol_litho1"] = out["viol_litho1"].fillna(False).astype(bool)
    out["viol_litho2"] = out["viol_litho2"].fillna(False).astype(bool)

    pt_excess = etch_pt_excess(log, route, etch_pt_mean, etch_pt_cv)
    out = out.merge(pt_excess[["lot_id", "pt_excess_etch"]], on="lot_id", how="left")
    out["pt_excess_etch"] = out["pt_excess_etch"].fillna(0.0)

    # Chamber (off-nominal tool) effect: only active when a label is given.
    if cfg.off_nominal_tool_label is not None:
        off_nominal1 = (out["litho1_tool"] == cfg.off_nominal_tool_label)
        off_nominal2 = (out["litho2_tool"] == cfg.off_nominal_tool_label)
        chamber_term = cfg.a_chamber * (off_nominal1 | off_nominal2).astype(float)
    else:
        chamber_term = 0.0

    p = (cfg.p_base
         + cfg.a_viol1 * out["viol_litho1"].astype(float)
         + cfg.a_viol2 * out["viol_litho2"].astype(float)
         + chamber_term
         + cfg.a_pt_excess * out["pt_excess_etch"])
    out["p_latent"] = p.clip(lower=0.0, upper=0.95)

    rng = np.random.default_rng(cfg.seed)
    out["defects"] = rng.binomial(WAFERS_PER_LOT, out["p_latent"].to_numpy())
    out["lot_yield"] = (WAFERS_PER_LOT - out["defects"]) / WAFERS_PER_LOT
    noise = rng.normal(0.0, metrology_sigma, size=len(out))
    out["metrology_reading"] = out["p_latent"].to_numpy() + noise

    return out[["lot_id", "viol_litho1", "viol_litho2", "litho1_tool", "litho2_tool",
                "pt_excess_etch", "p_latent", "defects", "lot_yield",
                "metrology_reading"]]
