"""
Synthetic fab-style production-line generator (Milestone M2, fab-ized in M7,
configurable dispatch policies added in M9 Stage A).

A transparent, hand-written discrete-event simulation (DES) of an open
multi-server queueing network shaped like a stylized wafer-fab flow. No external
DES library is used so that every modeling step is explicit and defensible.

The unit of flow is a **lot** - read it as a 25-wafer carrier (FOUP). The line is
a single stylized layer loop:

    CLEAN -> FURNACE -> DEPO -> LITHO -> ETCH -> LITHO -> IMPLANT -> METRO

Design choices (each is a deliberate, defendable trade-off):

- Open queueing network with FIFO multi-server stations. FIFO is the simplest
  baseline dispatch rule. Real fabs use far more complex dispatching (critical
  ratio, hot lots, setup avoidance); that is a documented limitation / future
  work, not an oversight.

- Re-entrant route: LITHO is visited twice (two mask layers of the same
  stylized loop). Re-entrance is the defining structural feature of
  semiconductor fabs and is the idea borrowed from the public SMT2020 testbed.
  It is also what makes the designed bottleneck loaded.

- One **batch tool**: FURNACE (diffusion/oxidation) processes up to
  ``batch_size`` lots in a single run - the second defining fab feature. Policy
  is greedy load-and-go: when a tool frees, it takes up to ``batch_size``
  waiting lots as one run (no minimum-batch or timeout policy; documented
  simplification). Run time is one processing-time draw (the first-loaded
  lot's draw), reflecting that a furnace recipe time is set by the recipe, not
  by how many lots are loaded.

- Lognormal processing times: positive and right-skewed, which matches real
  process-time behavior. The coefficient of variation (cv) controls variability;
  variability is what creates queueing (Kingman intuition), so WIP builds in
  front of the busiest station. FURNACE uses a lower cv (recipe-controlled).

- Poisson arrivals (exponential inter-arrival times): the standard memoryless
  arrival assumption for an open line.

- A single station: LITHO is engineered to have the least capacity headroom
  (highest utilization, ~0.85), matching the real-fab pattern where litho
  scanners are the most expensive tools and are run closest to saturation. By
  Theory of Constraints it sets line performance, so it is the ground-truth
  bottleneck recorded in metadata and later used to validate the M4 detector.

- Capacity accounting for batch tools uses **slot utilization**
  (see ``theoretical_utilization``): a furnace that looks "slow per operation"
  can still have ample capacity because each run carries several lots. This is
  the standard fab view of batch-tool capacity and is what the naive
  "longest-processing-time" bottleneck heuristic gets wrong (M4, Step 2).

The simulation is fully seeded for reproducibility.

Common Random Numbers (CRN) - M4 addition
------------------------------------------
For the M4 counterfactual we need a *paired* comparison: baseline vs "+1 tool at
station X" must face the SAME random inputs so the measured delta reflects only
the capacity change, not a different random stream. To support that, all
randomness can be pre-drawn into a ``RandomDraws`` table via ``draw_randoms()``
and passed to ``simulate(cfg, draws=...)``; the event loop then consumes the
table and calls no RNG at all.

- ``simulate(cfg)`` with ``draws=None`` samples lazily inside the event loop
  from an internal RNG seeded by ``cfg.seed``.
- ``simulate(cfg, draws)`` with an explicit table is fully deterministic (no
  RNG), which is what makes baseline-vs-baseline on the same table produce an
  exact zero delta - the sanity check that proves no hidden RNG source escapes
  the table.
- Batch stations consume the FIRST loaded lot's draw as the run time; the other
  members' draws at that step go unused. Usage may differ between scenarios,
  but the table itself depends only on the seed and the distributional config,
  which is all CRN pairing requires.

Tool-level bookkeeping and chamber offsets - M7 Stage A addition
-----------------------------------------------------------------
Later M7 stages (chamber matching, run-to-run comparison) need to know WHICH
physical tool served each operation and need the two LITHO tools to run with
slightly different means. Two additive, opt-in features support that:

- ``tool_id`` (always on): every event-log row now carries the specific tool
  that ran it, e.g. ``"LITHO-1"``. Assignment is deterministic bookkeeping
  (lowest-index free tool, see ``_ToolPool``) with NO new RNG draws anywhere,
  so it cannot perturb the draw table or the lazy-path RNG stream. The column
  is appended at the END of the schema so existing positional consumers are
  unaffected:
  ``lot_id, product_type, step_seq, station, queue_entry_time,
  process_start_time, process_complete_time, tool_id``.
- ``StationConfig.tool_offsets`` (opt-in, default ``None``): a per-tool
  multiplicative offset on mean processing time, applied by SCALING THE
  ALREADY-DRAWN processing time (``draw * tool_offsets[tool_idx]``) after it
  is read from the RNG or the CRN table. The draw itself is never altered, so
  a CRN baseline-vs-treatment pair stays exactly paired regardless of whether
  offsets are enabled. Default ``None`` multiplies by 1.0 everywhere, i.e.
  identical behavior to before this feature existed. Batch stations apply the
  offset of the tool that runs the batch to the first-loaded lot's draw (the
  same draw the batch already uses as its run time).

Configurable dispatch policies - M9 Stage A addition
-----------------------------------------------------
M9 compares dispatch policies (FIFO / EDD / critical ratio / queue-time-aware)
and a bottleneck-WIP release control against the locked FIFO baseline. Three
opt-in, additive mechanisms support that, all defaulting to the exact pre-M9
behavior:

- ``FactoryConfig.queue_discipline`` (default ``"fifo"``): the priority rule
  used when a tool frees and selects the next waiting lot(s). See
  ``_dispatch_order`` for the four disciplines and their tie-breaking rules.
  The "fifo" branch returns the station's pending list UNCHANGED - it is the
  same list object the pre-M9 code always consumed with ``pop(0)``, not just
  "priority math that happens to agree with FIFO" - so the default run is
  provably the original code path.
- ``FactoryConfig.flow_factor`` (default ``1.8``, see ``compute_due_date`` for
  the calibration): every lot gets a ``due_date`` at arrival, a deterministic
  function of arrival time and config (NO new RNG draws), appended to the
  ``lifecycle`` output the same way ``tool_id`` was appended to the event log.
- ``FactoryConfig.release_control`` (opt-in, default ``None``): a
  CONWIP-flavored pre-release pool that holds newly arrived lots out of CLEAN
  while LITHO WIP is at or above a threshold. See ``ReleaseControlConfig``.

Why dispatch-policy changes cannot desynchronize CRN pairing (draw indexing)
------------------------------------------------------------------------------
``RandomDraws.proc_times`` is indexed ``[lot_id][route_step]`` - i.e. by WHICH
LOT and WHICH POSITION IN ITS OWN ROUTE, never by queue position, dispatch
order, or wall-clock time. A dispatch policy only changes the ORDER in which
already-arrived, already-indexed lots are pulled off a station's pending list;
it never changes a lot's ``lot_id`` or which route step it is currently
requesting. Concretely, ``pt_for(lot, step, s)`` (plain path) / ``base_pt(lot,
step, s)`` (injected path) always read ``draws.proc_times[lot][step]`` - the
dispatch-order helper ``_dispatch_order`` is called BEFORE this lookup and
only decides WHICH pending entries become ``members`` for the next run; the
``(lot, step)`` key handed to the draw table afterward is whatever that
member's own dict says, untouched by the reordering.

Consequence for Stage B's paired policy comparison: running the SAME
``RandomDraws`` table (same seed) under two different ``queue_discipline``
values reads the exact same ``proc_times[lot][step]`` value for every lot at
every (station, visit) - the two runs merely consume those values in a
different ORDER as service starts happen at different simulated times. Two
runs can therefore differ in cycle time, WIP, and due-date performance (the
quantity M9 wants to compare) while remaining paired on the input randomness
(the quantity CRN requires to stay fixed for a valid comparison). This is
exactly the property ``dispatch_check.py`` GATE 4 asserts directly against the
event log.
"""

from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# M9 Stage A: the "queue_time_aware" dispatch policy needs the fixed post-LITHO
# window W (see queue_time.py docstring for its calibration). queue_time.py has
# no dependency back on this module, so importing it here is not circular. The
# path insert mirrors the pattern every *_check.py script already uses (see
# e.g. quality_check.py) since this repo has no package __init__.py wiring.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC / "quality") not in sys.path:
    sys.path.insert(0, str(_SRC / "quality"))
from queue_time import DEFAULT_WINDOW_HOURS as _POST_LITHO_WINDOW_HOURS  # noqa: E402


@dataclass
class StationConfig:
    """Configuration for one station (tool group).

    ``tool_offsets`` (M7, opt-in): a per-tool multiplicative offset on mean
    processing time, e.g. ``(1.03, 0.97)`` for a 2-tool station where tool 1
    runs 3% slower and tool 2 runs 3% faster than the station's nominal draw.
    When set, its length must equal ``n_tools``. Default ``None`` means every
    tool multiplies by 1.0 - identical behavior to before this feature existed.

    CRN safety: offsets are applied to the ALREADY-DRAWN processing time
    (``draw * tool_offsets[tool_idx]``), never folded into the lognormal
    parameters and never consuming an RNG call. The draw table from
    ``draw_randoms()`` is therefore unaffected by this field, and tool
    assignment (see ``simulate``'s ``_ToolPool``) is a deterministic
    lowest-index-free rule with no RNG involved either. Enabling offsets
    cannot desynchronize a CRN baseline-vs-treatment pair.
    """
    name: str
    pt_mean: float          # mean processing time per run (hours)
    n_tools: int            # number of parallel tools (servers)
    pt_cv: float = 0.5      # coefficient of variation of processing time
    batch_size: int = 1     # lots processed together per run (1 = serial tool)
    tool_offsets: tuple | None = None   # optional per-tool mean multiplier (M7)


@dataclass
class ReleaseControlConfig:
    """Bottleneck-WIP release control (M9 Stage A, opt-in, CONWIP-flavored).

    Stylized analogy to CONWIP (constant-WIP release): newly arrived lots do
    NOT enter CLEAN immediately. They sit in a pre-release pool and are
    released (one at a time, oldest first) only while a MEASURE of LITHO WIP
    is below ``litho_wip_threshold``.

    Two related quantities matter here, and it is important not to conflate
    them:

    - **Instantaneous LITHO WIP** (queued + in-process AT LITHO right now,
      both re-entrant visits) - this is what ``dispatch_check.py`` GATE 5
      measures and bounds, and what the module docstring / spec call
      "LITHO WIP".
    - **Committed WIP** (lots already released into the line but not yet past
      their SECOND LITHO visit) - this is what the release GATE ITSELF checks
      against the threshold, tracked internally as ``released_pending_litho``
      in ``simulate`` / ``_simulate_injected``.

    The gate must use committed WIP, not instantaneous WIP: a lot released
    into CLEAN takes several hours (CLEAN, FURNACE, DEPO) before it physically
    reaches LITHO, so checking instantaneous LITHO WIP alone would see "0
    lots at LITHO" for that whole transit window and release the ENTIRE
    pre-release pool in one burst every time LITHO happens to empty out - a
    release-timing bug, not a capacity control. Gating on committed WIP (lots
    already in flight toward/at LITHO) is the standard CONWIP fix: it caps how
    much work is loose in the CLEAN-to-LITHO span, which is what keeps
    instantaneous LITHO WIP from spiking well past the threshold later.

    This is deliberately simple (a single threshold on one station's WIP), not
    a full CONWIP card-count implementation with a global WIP cap; the name
    "CONWIP-flavored" in the docs reflects that simplification.
    """
    litho_wip_threshold: int   # release gate: hold lots while committed LITHO WIP >= this


@dataclass
class FactoryConfig:
    """Full factory / experiment configuration.

    M9 Stage A additions (both opt-in, both default to exact V1/M7/M8
    behavior):

    - ``queue_discipline``: priority rule used when a tool frees and picks the
      next lot from a station's waiting queue. ``"fifo"`` (default) is the
      ORIGINAL code path (earliest ``queue_entry_time`` first) - see
      ``_dispatch_order`` for the other disciplines and the byte-identity
      argument.
    - ``flow_factor``: multiplier used ONLY to compute each lot's due date at
      arrival (see ``compute_due_date``). Unused when no policy or gate reads
      due dates, but always computed since it is a deterministic function of
      already-known quantities (no new RNG draws).
    - ``release_control``: optional ``ReleaseControlConfig``; ``None``
      (default) disables it entirely and arrivals enter CLEAN immediately,
      exactly as before this feature existed.
    """
    stations: dict          # name -> StationConfig
    route: list             # ordered station names; repeats = re-entrant flow
    arrival_rate: float     # lots per hour (Poisson process)
    horizon_hours: float    # total simulated time
    warmup_hours: float     # initial period excluded from steady-state stats
    seed: int = 42
    product_type: str = "P1"
    queue_discipline: str = "fifo"           # M9: "fifo" | "edd" | "critical_ratio" | "queue_time_aware"
    flow_factor: float = 1.8                 # M9: due_date = arrival + flow_factor * raw_process_time_of_route
    release_control: ReleaseControlConfig | None = None  # M9: opt-in bottleneck-WIP release gate


@dataclass
class RandomDraws:
    """Pre-drawn randomness for one simulation replication (Common Random Numbers).

    Attributes
    ----------
    arrivals : list[float]
        Absolute arrival time of each lot, in arrival order. ``lot_id`` is the
        index into this list, so ``len(arrivals)`` fixes the lot count for the run.
    proc_times : list[list[float]]
        ``proc_times[lot_id][step]`` is the processing time (hours) that lot
        consumes at route position ``step``.

        IMPORTANT: indexing is by ROUTE STEP (visit order), NOT by station.
        The route is re-entrant:
        ``["CLEAN","FURNACE","DEPO","LITHO","ETCH","LITHO","IMPLANT","METRO"]``
        visits LITHO twice, at step 3 and step 5. Those are two INDEPENDENT
        draws, ``proc_times[lot][3]`` and ``proc_times[lot][5]``. Because the
        pairing is by step, baseline and any "+1 tool" treatment consume the
        exact same two LITHO draws in the exact same order - a re-entrant
        station cannot get its paired draws mis-aligned. Every lot traverses
        the full route exactly once (no rework in this model), so
        ``len(proc_times[lot]) == len(route)`` and the table depends only on
        the seed and the distributional config, never on ``n_tools`` or
        ``batch_size``. At a batch station only the first-loaded lot's draw is
        consumed as the run time; unused draws are harmless.
    """
    arrivals: list           # arrivals[lot_id] -> arrival time
    proc_times: list         # proc_times[lot_id][step] -> processing hours


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert a target mean and CV into lognormal (mu, sigma) parameters."""
    sigma2 = math.log(1.0 + cv ** 2)
    mu = math.log(mean) - sigma2 / 2.0
    return mu, math.sqrt(sigma2)


class _ToolPool:
    """Deterministic tool-index bookkeeping for one station (M7).

    Tracks which of a station's ``n_tools`` parallel tools are free/busy and
    hands out the LOWEST-INDEX free tool on each acquisition (stable, greedy
    rule - ties are impossible since indices are unique). Indices are 0-based
    internally and rendered as 1-based ``"{station}-{idx+1}"`` labels for the
    event log (e.g. ``"LITHO-1"``, ``"LITHO-2"``).

    This is pure bookkeeping: no random draws are made or consumed here, so
    adding tool_id tracking cannot perturb the RNG stream on the lazy path or
    the draw table on the CRN path. A run's tool is acquired at dispatch and
    released at completion, mirroring the existing free/busy counters exactly
    (this class only adds identity information on top of the same transitions).
    """

    def __init__(self, station: str, n_tools: int):
        self.station = station
        self._free_idx = list(range(n_tools))  # kept sorted -> lowest index first

    def acquire(self) -> int:
        idx = min(self._free_idx)
        self._free_idx.remove(idx)
        return idx

    def release(self, idx: int) -> None:
        self._free_idx.append(idx)

    def label(self, idx: int) -> str:
        return f"{self.station}-{idx + 1}"


# --------------------------------------------------------------------------- #
# Anomaly injection primitives (M5, extended M8)
# --------------------------------------------------------------------------- #
# These are the injection primitives the simulator interprets, so they live with
# the simulator (monitoring/ depends on generator, not the other way round). Each
# anomaly is a DETERMINISTIC function of time layered on top of the same base
# draws - the draw table is never mutated, so CRN pairing with a clean twin holds.
# Every anomaly carries an explicit [t_start, t_end] window = the ground truth.
#
# Contract used by simulate()'s injection path:
#   tools_delta(station, t)   -> int   change to effective n_tools at (station, t)
#   pt_multiplier(station, t) -> float multiplier on processing time at (station, t)
#   extra_arrivals(cfg)       -> list[(arrival_time, [proc_time per route step])]
#   boundaries()              -> list[float] times to re-evaluate dispatch
#   label()                   -> dict describing the injected window (for meta)
# When no anomaly applies, tools_delta=0 / pt_multiplier=1.0 / extra_arrivals=[],
# i.e. the identity - so simulate(..., anomalies=[]) equals the un-injected run.
#
# OEE reading (M5): a BreakdownAnomaly is an **Availability loss** (tools
# offline); a DegradationAnomaly is a **Performance loss** (running slower than
# the ideal rate). The two standard equipment-loss categories of OEE.
#
# M8 addition: ScheduledDowntimeAnomaly reuses BreakdownAnomaly's exact
# capacity-reduction mechanics (same tools_delta contract) but is labeled
# "scheduled_pm" instead of "breakdown", so the E10 tool-state layer can tell
# planned maintenance apart from an unplanned failure. Injection metadata for
# ALL anomalies (including this one) is returned in ``meta["anomalies"]`` by
# ``_simulate_injected`` - see that function's docstring - so downstream
# consumers never need to re-derive windows from the event log.

@dataclass
class BreakdownAnomaly:
    """Capacity mask: reduce a station's effective n_tools during a window.

    Models an availability drop (tools offline) - an OEE **Availability loss**.
    Arrival and processing draws are untouched; only the number of serving
    tools changes, restored at ``t_end``.
    """
    station: str
    t_start: float
    t_end: float
    tools_removed: int = 1

    def tools_delta(self, station: str, t: float) -> int:
        if station == self.station and self.t_start <= t < self.t_end:
            return -self.tools_removed
        return 0

    def pt_multiplier(self, station: str, t: float) -> float:
        return 1.0

    def extra_arrivals(self, cfg) -> list:
        return []

    def boundaries(self) -> list:
        return [self.t_start, self.t_end]

    def label(self) -> dict:
        return {"type": "breakdown", "station": self.station,
                "t_start": self.t_start, "t_end": self.t_end,
                "tools_removed": self.tools_removed}


@dataclass
class ScheduledDowntimeAnomaly:
    """Capacity mask: reduce a station's effective n_tools during a window,
    semantically labeled SCHEDULED (planned PM) rather than unscheduled (M8).

    Mechanically this reuses the exact same capacity-reduction path as
    ``BreakdownAnomaly`` (``tools_delta`` masks ``effective_capacity`` in
    ``_simulate_injected``); the only difference is the ``label()`` type
    string, which downstream consumers (the M8 E10 tool-state layer) read to
    tell a planned maintenance window apart from an unplanned failure. Arrival
    and processing draws are untouched, exactly like a breakdown.

    Tool convention (M8): ``_simulate_injected`` reduces a station's capacity
    COUNT, not a specific tool index (the dispatch loop only ever asks "how
    many tools are free", never "which ones"), so no single physical tool is
    mechanically "the one that is down". The E10 state-builder
    (``src/equipment/e10_states.py``) therefore ADOPTS A CONVENTION: the
    HIGHEST-index tool of the station is the one attributed as down for the
    window (e.g. a 2-tool station's "-2" tool). This is a labeling choice for
    the state layer, not a simulation mechanic; it is documented again at the
    point of use.
    """
    station: str
    t_start: float
    t_end: float
    tools_removed: int = 1

    def tools_delta(self, station: str, t: float) -> int:
        if station == self.station and self.t_start <= t < self.t_end:
            return -self.tools_removed
        return 0

    def pt_multiplier(self, station: str, t: float) -> float:
        return 1.0

    def extra_arrivals(self, cfg) -> list:
        return []

    def boundaries(self) -> list:
        return [self.t_start, self.t_end]

    def label(self) -> dict:
        return {"type": "scheduled_pm", "station": self.station,
                "t_start": self.t_start, "t_end": self.t_end,
                "tools_removed": self.tools_removed}


@dataclass
class DegradationAnomaly:
    """Deterministic slow drift: processing time ramps up over a window.

    Models a tool running slower than its ideal rate - an OEE **Performance
    loss**. Effective processing time = base_draw * (1 + alpha * (t - t_onset))
    at the station for t in [t_onset, t_end]. The multiplier is a pure function
    of time (no extra random draws - that would break CRN). After t_end, back
    to normal.
    """
    station: str
    t_onset: float
    t_end: float
    alpha: float          # fractional processing-time increase per hour

    def tools_delta(self, station: str, t: float) -> int:
        return 0

    def pt_multiplier(self, station: str, t: float) -> float:
        if station == self.station and self.t_onset <= t <= self.t_end:
            return 1.0 + self.alpha * (t - self.t_onset)
        return 1.0

    def extra_arrivals(self, cfg) -> list:
        return []

    def boundaries(self) -> list:
        return [self.t_onset, self.t_end]

    def label(self) -> dict:
        return {"type": "degradation", "station": self.station,
                "t_start": self.t_onset, "t_end": self.t_end, "alpha": self.alpha}


@dataclass
class DemandSurgeAnomaly:
    """Extra arrivals during a window, drawn from a separate seeded stream.

    The base draw table is untouched; the surge adds lots (disjoint lot_ids) whose
    inter-arrival and per-step processing times come from this anomaly's own seed.
    """
    t_start: float
    t_end: float
    extra_rate: float     # additional lots per hour
    seed: int = 7

    def tools_delta(self, station: str, t: float) -> int:
        return 0

    def pt_multiplier(self, station: str, t: float) -> float:
        return 1.0

    def extra_arrivals(self, cfg) -> list:
        rng = np.random.default_rng(self.seed)
        params = {s: _lognormal_params(st.pt_mean, st.pt_cv)
                  for s, st in cfg.stations.items()}
        out = []
        t = self.t_start
        while True:
            t += rng.exponential(1.0 / self.extra_rate)
            if t >= self.t_end:
                break
            pts = [float(rng.lognormal(*params[cfg.route[step]]))
                   for step in range(len(cfg.route))]
            out.append((t, pts))
        return out

    def boundaries(self) -> list:
        return [self.t_start, self.t_end]

    def label(self) -> dict:
        return {"type": "demand_surge", "t_start": self.t_start,
                "t_end": self.t_end, "extra_rate": self.extra_rate}


def theoretical_utilization(cfg: FactoryConfig) -> dict:
    """
    Design-time (slot) utilization per station:

        rho_s = arrival_rate * visits_s * pt_mean_s / (n_tools_s * batch_size_s)

    For serial tools (batch_size 1) this is the classic queueing utilization.
    For batch tools it is **slot utilization** - work arriving per hour divided
    by lot-slots servable per hour - the standard fab measure of batch-tool
    capacity. It assumes full batches, so a greedy loading policy will show a
    higher busy-time fraction than this number; that is exactly why busy-time
    alone misreads batch tools and slot utilization is the capacity view.

    This is the planned load and identifies the intended bottleneck before any
    simulation runs. The DES should reproduce this ordering empirically.
    """
    visits = {s: cfg.route.count(s) for s in cfg.stations}
    rho = {}
    for s, st in cfg.stations.items():
        rho[s] = cfg.arrival_rate * visits[s] * st.pt_mean / (st.n_tools * st.batch_size)
    return rho


# --------------------------------------------------------------------------- #
# Due dates and dispatch-priority policies (M9 Stage A)
# --------------------------------------------------------------------------- #
def raw_process_time_of_route(cfg: FactoryConfig) -> float:
    """Sum of each route step's MEAN processing time (hours), config-only.

    This is a fixed property of ``cfg.route`` / ``cfg.stations`` (no draws, no
    simulation state), so it can be computed once and reused as the basis for
    every lot's due date. For the locked default route (CLEAN, FURNACE, DEPO,
    LITHO, ETCH, LITHO, IMPLANT, METRO with the default station means) this is
    10.0 hours.
    """
    return float(sum(cfg.stations[s].pt_mean for s in cfg.route))


def compute_due_date(arrival_time: float, cfg: FactoryConfig,
                      raw_route_time: float | None = None) -> float:
    """Due date assigned at arrival: due = arrival_time + flow_factor * raw_route_time.

    Deterministic from the arrival time and config only - NO new RNG draws, so
    computing due dates cannot perturb the draw table (CRN path) or the lazy
    RNG stream. ``raw_route_time`` may be passed in to avoid recomputing the
    route sum per lot; if omitted it is derived from ``cfg``.

    Calibration of ``cfg.flow_factor`` (default 1.8)
    -------------------------------------------------
    Computed ONCE against the default-seed baseline (``default_config()`` +
    ``simulate(cfg)``, seed 42, FIFO, no anomalies, no release control),
    restricted to the steady-state window ``[warmup_hours, horizon_hours]``
    (1280 lots). Sweeping flow_factor over the realized cycle-time
    distribution (mean 14.14 h, p50 13.53 h, p90 19.30 h) gives:

        flow_factor=1.6  on_time_rate=0.731
        flow_factor=1.8  on_time_rate=0.858   <- chosen default
        flow_factor=2.0  on_time_rate=0.926
        flow_factor=2.2  on_time_rate=0.966

    1.8 was picked because it lands mid-band in the 70-95% "discriminating"
    range requested for M9 (on-time rate that can visibly move between
    dispatch policies, rather than saturating near 0% or 100%). This is a
    fixed config constant, like ``queue_time.DEFAULT_WINDOW_HOURS`` - it is
    not recalibrated automatically if the generator changes; a deliberate
    re-calibration would need to be documented again at this docstring.
    """
    if raw_route_time is None:
        raw_route_time = raw_process_time_of_route(cfg)
    return arrival_time + cfg.flow_factor * raw_route_time


#: Stations immediately following a LITHO visit in the locked route (ETCH
#: follows the first LITHO visit at step 3; IMPLANT follows the second at step
#: 5). Used by the "queue_time_aware" policy below.
_POST_LITHO_STATIONS = ("ETCH", "IMPLANT")


def _remaining_work(step: int, route: list, stations: dict) -> float:
    """Sum of MEAN processing times for route steps NOT YET STARTED, inclusive
    of the current pending step. Used by the "critical_ratio" policy as the
    denominator. Config-only (no draws), matching ``raw_process_time_of_route``.
    """
    return float(sum(stations[route[i]].pt_mean for i in range(step, len(route))))


def _dispatch_order(pending: list, s: str, now: float, cfg: FactoryConfig,
                     raw_route_time: float) -> list:
    """Return ``pending`` (a station's waiting-lot dict list) reordered by
    ``cfg.queue_discipline`` for the NEXT run's member selection. Does not
    mutate the input list.

    Each pending entry is a dict with keys ``lot``, ``step``, ``qentry``,
    ``due`` (due date, always present - see ``compute_due_date``).

    - "fifo" (default): returns ``pending`` UNCHANGED - this is the identical
      list/order the pre-M9 code always used (``pending[s].pop(0)`` after
      ``append``), so the FIFO path is not merely "equivalent priority math",
      it is literally a no-op over the original list. This is the byte-identity
      guarantee gate 1 in dispatch_check.py exercises.
    - "edd": earliest due date first; ties broken by queue_entry_time (stable
      sort keeps original FIFO order among equal due dates, which are equal-
      arrival-basis lots since due dates are a fixed offset of arrival time).
    - "critical_ratio": ascending (due - now) / remaining_work, ties broken by
      queue_entry_time. A smaller ratio means less slack per unit of work
      left, i.e. more urgent. ``remaining_work`` is the MEAN processing time
      of steps from the current pending step through the end of route (see
      ``_remaining_work``) - a deterministic, config-only quantity, not a
      random draw, so this policy needs no RNG either.
    - "queue_time_aware": ONLY at ETCH and IMPLANT (the two stations that
      immediately follow a LITHO visit, see ``_POST_LITHO_STATIONS``), lots
      are prioritized by LEAST remaining post-litho window slack:
      ``slack = W - (now - queue_entry_time)`` (queue_entry_time here IS the
      LITHO completion time, since the lot enters this station's queue the
      instant it leaves LITHO), ascending (least slack = closest to violating
      the window W = 0.4102 h from queue_time.DEFAULT_WINDOW_HOURS is served
      first); ties broken by queue_entry_time. At every OTHER station this
      policy falls back to plain FIFO (unchanged pending order).

    Batch stations: this function only decides WHICH waiting lots load next;
    run time is still the first-loaded (post-reorder) lot's draw, unchanged
    batch semantics (see module docstring / try_dispatch).
    """
    discipline = cfg.queue_discipline
    if discipline == "fifo":
        return pending

    if discipline == "edd":
        return sorted(pending, key=lambda p: (p["due"], p["qentry"]))

    if discipline == "critical_ratio":
        def cr_key(p):
            remaining = _remaining_work(p["step"], cfg.route, cfg.stations)
            ratio = (p["due"] - now) / remaining if remaining > 0 else float("-inf")
            return (ratio, p["qentry"])
        return sorted(pending, key=cr_key)

    if discipline == "queue_time_aware":
        if s not in _POST_LITHO_STATIONS:
            return pending

        def slack_key(p):
            slack = _POST_LITHO_WINDOW_HOURS - (now - p["qentry"])
            return (slack, p["qentry"])
        return sorted(pending, key=slack_key)

    raise ValueError(f"Unknown queue_discipline: {discipline!r}")


def draw_randoms(cfg: FactoryConfig, seed: int) -> RandomDraws:
    """Pre-draw all randomness for one replication into a reusable table (CRN).

    The returned table depends only on ``seed`` and the *distributional* config
    (arrival_rate, route, and each station's pt_mean / pt_cv). It does NOT depend
    on ``n_tools`` or ``batch_size``. That is the whole point: generate one table
    per replication, then run the baseline and every "+1 tool" scenario against
    that SAME table so they face identical arrivals and identical per-visit
    processing times, and the only thing that varies is capacity.

    Draw order (single RNG stream, documented for reproducibility):
      1. Inter-arrival times, accumulated until ``horizon_hours`` (same rule the
         ``draws=None`` path uses to schedule arrivals).
      2. Then, per lot in arrival order, one processing-time draw per route step,
         in step order. See ``RandomDraws.proc_times`` for the by-step indexing
         and why it keeps re-entrant LITHO paired correctly.
    """
    rng = np.random.default_rng(seed)

    # 1) Arrivals - identical generation rule to the lazy path.
    arrivals: list = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / cfg.arrival_rate)
        if t >= cfg.horizon_hours:
            break
        arrivals.append(t)

    # 2) Processing times, indexed by (lot, route step). LITHO at steps 3 and 5
    #    gets two independent draws here; both are reused by baseline and
    #    treatment.
    lognorm_params = {
        s: _lognormal_params(st.pt_mean, st.pt_cv)
        for s, st in cfg.stations.items()
    }
    proc_times: list = []
    for _ in arrivals:
        lot_pts = []
        for step, s in enumerate(cfg.route):
            mu, sigma = lognorm_params[s]
            lot_pts.append(float(rng.lognormal(mu, sigma)))
        proc_times.append(lot_pts)

    return RandomDraws(arrivals=arrivals, proc_times=proc_times)


def simulate(cfg: FactoryConfig, draws: RandomDraws | None = None,
             anomalies: list | None = None):
    """
    Run the discrete-event simulation (batch-aware).

    Parameters
    ----------
    cfg : FactoryConfig
        Factory / experiment configuration.
    draws : RandomDraws | None
        If ``None`` (default), randomness is sampled lazily inside the event
        loop from an internal RNG seeded by ``cfg.seed``. If a ``RandomDraws``
        table is provided (Common Random Numbers), the loop consumes it and
        calls NO RNG, so the run is fully deterministic and paired against any
        other run that uses the same table.
    anomalies : list | None
        If ``None`` or empty (default), the plain un-injected path below runs.
        If anomalies are given, control passes to ``_simulate_injected`` (M5),
        which layers the anomalies' deterministic, time-based transforms on top
        of the same draws.

    Batch semantics (both paths): a station with ``batch_size`` B loads up to B
    waiting lots as ONE run on one tool (greedy load-and-go). Queue discipline
    (M9) decides WHICH waiting lots load first; the run's processing time is
    still the first-loaded (post-reorder) lot's draw, and all members share
    the same process_start / process_complete and then advance individually.

    M9 Stage A additions (all opt-in, all default to the exact pre-M9 code
    path - see each dataclass/helper's own docstring for the byte-identity
    argument):

    - ``cfg.queue_discipline`` ("fifo" default): reorders each station's
      waiting-lot list at dispatch time via ``_dispatch_order``.
    - ``cfg.flow_factor``: every lot gets a ``due_date`` at arrival (see
      ``compute_due_date``), appended to the returned ``lifecycle`` frame.
      This is a pure function of arrival time and config, so it is always
      computed (no RNG, cannot affect determinism or CRN pairing).
    - ``cfg.release_control`` (``None`` default): optional pre-release pool
      gating arrivals into CLEAN by LITHO WIP (see ``ReleaseControlConfig``).

    Returns
    -------
    log : pd.DataFrame
        One row per completed operation (batch members get one row each):
        [lot_id, product_type, step_seq, station,
         queue_entry_time, process_start_time, process_complete_time, tool_id]
    lifecycle : pd.DataFrame
        One row per lot: [lot_id, arrival_time, completion_time, due_date].
    meta : dict
        Configuration echo + ground-truth bottleneck + per-station capacity.
    """
    if anomalies:
        return _simulate_injected(cfg, draws, anomalies)

    # RNG exists ONLY on the lazy (draws=None) path. On the CRN path it stays
    # None and must never be touched - if it were, baseline-vs-baseline on one
    # table would not be an exact zero and the CRN sanity check would catch it.
    rng = np.random.default_rng(cfg.seed) if draws is None else None

    free = {s: st.n_tools for s, st in cfg.stations.items()}   # free tools per station
    pending = {s: [] for s in cfg.stations}                    # per-station queues
    tools = {s: _ToolPool(s, st.n_tools) for s, st in cfg.stations.items()}  # M7
    rows = []
    arrivals: dict[int, float] = {}
    completions: dict[int, float] = {}
    due_dates: dict[int, float] = {}

    raw_route_time = raw_process_time_of_route(cfg)     # M9: fixed, config-only
    litho_steps = {i for i, st in enumerate(cfg.route) if st == "LITHO"}  # M9 release control
    last_litho_step = max(litho_steps) if litho_steps else None

    # M9 release control: pre-release pool. Disabled (release_control is None)
    # means every arriving lot is requested into CLEAN immediately, exactly
    # the pre-M9 behavior.
    release_pool: list = []                              # [lot_id] oldest-first
    # Committed WIP: lots already released but not yet past their SECOND
    # LITHO visit. This, not instantaneous LITHO WIP, is what gates release -
    # see ReleaseControlConfig's docstring for why (transit-time burst bug).
    released_pending_litho = 0

    def litho_wip(now) -> int:
        """Lots queued OR in-process at LITHO, both re-entrant visits (M9).

        This is the INSTANTANEOUS measure reported/bounded by
        dispatch_check.py GATE 5. It is NOT what gates release (see
        ``released_pending_litho`` above / ``ReleaseControlConfig`` docstring).
        """
        queued = sum(1 for e in pending.get("LITHO", []) if e["step"] in litho_steps)
        in_proc = sum(1 for _, _, kind, payload in heap
                      if kind == "complete"
                      and cfg.route[payload["members"][0]["step"]] == "LITHO")
        return queued + in_proc

    heap: list = []
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(heap, (t, seq, kind, payload))
        seq += 1

    def sample_pt(s):
        st = cfg.stations[s]
        mu, sigma = _lognormal_params(st.pt_mean, st.pt_cv)
        return float(rng.lognormal(mu, sigma))

    def pt_for(lot, step, s):
        """Processing time for the run led by ``lot`` at route position ``step``.

        Lazy path (draws=None): draw at run start. CRN path: read the pre-drawn
        value indexed by (lot, step) - by step, so re-entrant LITHO (steps 3
        and 5) stays paired. Queue discipline only reorders WHICH lot leads a
        run; it never changes which (lot, step) index is read, so draws stay
        paired across policies too (see module docstring, "Draw indexing").
        """
        if draws is None:
            return sample_pt(s)
        return draws.proc_times[lot][step]

    def try_dispatch(s, now):
        """Start as many runs as free tools allow; each takes up to batch_size."""
        B = cfg.stations[s].batch_size
        offsets = cfg.stations[s].tool_offsets
        while free[s] > 0 and pending[s]:
            ordered = _dispatch_order(pending[s], s, now, cfg, raw_route_time)
            k = min(B, len(ordered))
            chosen = ordered[:k]
            for c in chosen:
                pending[s].remove(c)
            members = chosen
            free[s] -= 1
            tool_idx = tools[s].acquire()
            lead = members[0]
            pt = pt_for(lead["lot"], lead["step"], s)
            if offsets is not None:
                pt *= offsets[tool_idx]          # M7: post-draw scale only, no RNG
            push(now + pt, "complete",
                 {"members": members, "start": now, "tool_idx": tool_idx})

    def request(lot, step, now):
        """Lot requests the station for this route step (queue + dispatch)."""
        s = cfg.route[step]
        pending[s].append({"lot": lot, "step": step, "qentry": now,
                            "due": due_dates[lot]})
        try_dispatch(s, now)

    def try_release(now):
        """M9: release oldest pooled lots into CLEAN while COMMITTED WIP
        (``released_pending_litho``, not instantaneous LITHO WIP - see
        ``ReleaseControlConfig`` docstring) is below the configured
        threshold. No-op (pool always empty) when release_control is None, so
        this cannot change behavior when the feature is off.
        """
        nonlocal released_pending_litho
        if cfg.release_control is None:
            return
        threshold = cfg.release_control.litho_wip_threshold
        while release_pool and released_pending_litho < threshold:
            lot = release_pool.pop(0)
            released_pending_litho += 1
            request(lot, 0, now)

    def arrive(lot_id, at):
        arrivals[lot_id] = at
        due_dates[lot_id] = compute_due_date(at, cfg, raw_route_time)
        if cfg.release_control is None:
            request(lot_id, 0, at)
        else:
            release_pool.append(lot_id)
            try_release(at)

    # Schedule arrivals up front.
    if draws is None:
        # Lazy: Poisson arrivals sampled from the internal RNG.
        t, lot_id = 0.0, 0
        while True:
            t += rng.exponential(1.0 / cfg.arrival_rate)
            if t >= cfg.horizon_hours:
                break
            push(t, "arrive", {"lot": lot_id})
            lot_id += 1
    else:
        # CRN: arrivals come straight from the pre-drawn table.
        for lot_id, at in enumerate(draws.arrivals):
            push(at, "arrive", {"lot": lot_id})

    # Event loop.
    while heap:
        now, _, kind, p = heapq.heappop(heap)

        if kind == "arrive":
            arrive(p["lot"], now)
            continue

        # kind == "complete" - one run finishes; all members complete together.
        members = p["members"]
        s = cfg.route[members[0]["step"]]
        tool_label = tools[s].label(p["tool_idx"])
        for m in members:
            rows.append({
                "lot_id": m["lot"],
                "product_type": cfg.product_type,
                "step_seq": m["step"],
                "station": s,
                "queue_entry_time": m["qentry"],
                "process_start_time": p["start"],
                "process_complete_time": now,
                "tool_id": tool_label,
            })
        free[s] += 1
        tools[s].release(p["tool_idx"])

        # A tool just freed: pull the next waiting run at this station.
        try_dispatch(s, now)

        # Advance each completed lot to its next route step (or finish).
        for m in members:
            nstep = m["step"] + 1
            if nstep < len(cfg.route):
                request(m["lot"], nstep, now)
            else:
                completions[m["lot"]] = now
            # M9: a lot passing its SECOND (last) LITHO visit frees committed
            # release-pool headroom - see ReleaseControlConfig docstring for
            # why this, not instantaneous LITHO WIP, is the release gate.
            if cfg.release_control is not None and m["step"] == last_litho_step:
                released_pending_litho -= 1
                try_release(now)

    log = (pd.DataFrame(rows)
           .sort_values(["lot_id", "step_seq"])
           .reset_index(drop=True))

    lifecycle = pd.DataFrame({
        "lot_id": list(arrivals.keys()),
        "arrival_time": list(arrivals.values()),
    })
    lifecycle["completion_time"] = lifecycle["lot_id"].map(completions)
    lifecycle["due_date"] = lifecycle["lot_id"].map(due_dates)   # M9, append-only

    rho = theoretical_utilization(cfg)
    bottleneck = max(rho, key=rho.get)
    meta = {
        "seed": cfg.seed,
        "arrival_rate": cfg.arrival_rate,
        "horizon_hours": cfg.horizon_hours,
        "warmup_hours": cfg.warmup_hours,
        "route": cfg.route,
        "stations": {
            s: {"n_tools": st.n_tools, "batch_size": st.batch_size,
                "pt_mean": st.pt_mean, "pt_cv": st.pt_cv}
            for s, st in cfg.stations.items()
        },
        "theoretical_utilization": rho,
        "ground_truth_bottleneck": bottleneck,
        "queue_discipline": cfg.queue_discipline,
        "flow_factor": cfg.flow_factor,
    }
    return log, lifecycle, meta


def _simulate_injected(cfg: FactoryConfig, draws: RandomDraws | None, anomalies: list):
    """Injection-aware DES (M5). Only reached when ``anomalies`` is non-empty.

    Differs from the un-injected path in three isolated ways, all identity when no
    anomaly is active:

      * capacity is time-varying: a ``busy[s]`` counter is checked against
        ``effective_capacity(s, now) = n_tools + sum(tools_delta)`` instead of a
        static ``free[s]`` (lets a breakdown mask reduce serving tools);
      * run times are scaled by ``prod(pt_multiplier(s, now))`` at service
        start (lets a degradation ramp slow a station);
      * extra arrivals from demand-surge anomalies are scheduled from their own
        seeded stream, with lot_ids in a disjoint range so base draws are untouched.

    Batch semantics are identical to the plain path: up to ``batch_size`` lots
    per run, run time = first-loaded lot's draw (times the multiplier).

    Boundary events are scheduled at every anomaly window edge so that when a
    breakdown ends (capacity restored) any waiting lots are re-dispatched even if
    no completion happens to fire at that instant.

    M9 Stage A additions: identical mechanism to the plain path (see
    ``simulate``'s docstring) - ``cfg.queue_discipline`` reorders each
    station's waiting-lot list via the same ``_dispatch_order`` helper,
    ``due_date`` is computed for every lot (base and surge) and appended to
    ``lifecycle``, and ``cfg.release_control`` gates arrivals the same way.
    All default to the pre-M9 identity behavior.
    """
    rng = np.random.default_rng(cfg.seed) if draws is None else None

    base_tools = {s: st.n_tools for s, st in cfg.stations.items()}
    busy = {s: 0 for s in cfg.stations}                        # tools in service
    pending = {s: [] for s in cfg.stations}                    # per-station queues
    # M7: pool sized to the station's full (nominal) tool count. A breakdown only
    # limits how many of these indices may be acquired concurrently (via
    # effective_capacity below); it does not change which indices exist, so
    # tool_id labeling and offsets stay well-defined even during a breakdown window.
    tools = {s: _ToolPool(s, st.n_tools) for s, st in cfg.stations.items()}
    rows = []
    arrivals: dict[int, float] = {}
    completions: dict[int, float] = {}
    extra_pts: dict[int, list] = {}                            # surge lot -> proc times
    due_dates: dict[int, float] = {}

    raw_route_time = raw_process_time_of_route(cfg)     # M9: fixed, config-only
    litho_steps = {i for i, st in enumerate(cfg.route) if st == "LITHO"}  # M9 release control
    last_litho_step = max(litho_steps) if litho_steps else None
    release_pool: list = []                              # M9: [lot_id] oldest-first
    # Committed WIP gate - see ReleaseControlConfig docstring (transit-time
    # burst bug this avoids) and simulate()'s identical mechanism.
    released_pending_litho = 0

    def litho_wip(now) -> int:
        """Lots queued OR in-process at LITHO, both re-entrant visits (M9).
        Instantaneous measure only - see ``released_pending_litho`` above."""
        queued = sum(1 for e in pending.get("LITHO", []) if e["step"] in litho_steps)
        in_proc = sum(1 for _, _, kind, payload in heap
                      if kind == "complete"
                      and cfg.route[payload["members"][0]["step"]] == "LITHO")
        return queued + in_proc

    heap: list = []
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(heap, (t, seq, kind, payload))
        seq += 1

    def sample_pt(s):
        st = cfg.stations[s]
        mu, sigma = _lognormal_params(st.pt_mean, st.pt_cv)
        return float(rng.lognormal(mu, sigma))

    def base_pt(lot, step, s):
        if lot in extra_pts:                 # surge lot carries its own draws
            return extra_pts[lot][step]
        if draws is None:
            return sample_pt(s)
        return draws.proc_times[lot][step]

    def effective_capacity(s, now):
        delta = sum(a.tools_delta(s, now) for a in anomalies)
        return max(0, base_tools[s] + delta)

    def pt_multiplier(s, now):
        m = 1.0
        for a in anomalies:
            m *= a.pt_multiplier(s, now)
        return m

    def try_dispatch(s, now):
        # Start as many runs as the (possibly reduced) capacity allows;
        # each run takes up to batch_size waiting lots, ordered by queue_discipline.
        B = cfg.stations[s].batch_size
        offsets = cfg.stations[s].tool_offsets
        while pending[s] and busy[s] < effective_capacity(s, now):
            ordered = _dispatch_order(pending[s], s, now, cfg, raw_route_time)
            k = min(B, len(ordered))
            members = ordered[:k]
            for c in members:
                pending[s].remove(c)
            busy[s] += 1
            tool_idx = tools[s].acquire()
            lead = members[0]
            pt = base_pt(lead["lot"], lead["step"], s) * pt_multiplier(s, now)
            if offsets is not None:
                pt *= offsets[tool_idx]          # M7: post-draw scale only, no RNG
            push(now + pt, "complete",
                 {"members": members, "start": now, "tool_idx": tool_idx})

    def request(lot, step, now):
        s = cfg.route[step]
        pending[s].append({"lot": lot, "step": step, "qentry": now,
                            "due": due_dates[lot]})
        try_dispatch(s, now)

    def try_release(now):
        """M9: same release-pool mechanism as the plain path (see simulate()) -
        gates on committed WIP (``released_pending_litho``), not instantaneous
        LITHO WIP."""
        nonlocal released_pending_litho
        if cfg.release_control is None:
            return
        threshold = cfg.release_control.litho_wip_threshold
        while release_pool and released_pending_litho < threshold:
            lot = release_pool.pop(0)
            released_pending_litho += 1
            request(lot, 0, now)

    def arrive(lot_id, at):
        arrivals[lot_id] = at
        due_dates[lot_id] = compute_due_date(at, cfg, raw_route_time)
        if cfg.release_control is None:
            request(lot_id, 0, at)
        else:
            release_pool.append(lot_id)
            try_release(at)

    # Base arrivals (identical source to the plain path).
    if draws is None:
        t, lot_id = 0.0, 0
        while True:
            t += rng.exponential(1.0 / cfg.arrival_rate)
            if t >= cfg.horizon_hours:
                break
            push(t, "arrive", {"lot": lot_id})
            lot_id += 1
    else:
        for lot_id, at in enumerate(draws.arrivals):
            push(at, "arrive", {"lot": lot_id})

    # Extra arrivals from demand-surge anomalies (disjoint lot_ids, own draws).
    next_extra = 1_000_000
    for a in anomalies:
        for at, pts in a.extra_arrivals(cfg):
            extra_pts[next_extra] = pts
            push(at, "arrive", {"lot": next_extra})
            next_extra += 1

    # Boundary events so restored capacity re-triggers dispatch at window edges.
    for a in anomalies:
        for tb in a.boundaries():
            if 0 <= tb <= cfg.horizon_hours:
                push(tb, "boundary", {})

    # Event loop.
    while heap:
        now, _, kind, p = heapq.heappop(heap)

        if kind == "arrive":
            arrive(p["lot"], now)
            continue

        if kind == "boundary":
            for s in cfg.stations:
                try_dispatch(s, now)
            continue

        # kind == "complete" - one run finishes; all members complete together.
        members = p["members"]
        s = cfg.route[members[0]["step"]]
        tool_label = tools[s].label(p["tool_idx"])
        for m in members:
            rows.append({
                "lot_id": m["lot"],
                "product_type": cfg.product_type,
                "step_seq": m["step"],
                "station": s,
                "queue_entry_time": m["qentry"],
                "process_start_time": p["start"],
                "process_complete_time": now,
                "tool_id": tool_label,
            })
        busy[s] -= 1
        tools[s].release(p["tool_idx"])
        try_dispatch(s, now)                 # pull next waiting run

        for m in members:
            nstep = m["step"] + 1
            if nstep < len(cfg.route):
                request(m["lot"], nstep, now)
            else:
                completions[m["lot"]] = now
            # M9: a lot passing its SECOND (last) LITHO visit frees committed
            # release-pool headroom (see ReleaseControlConfig docstring).
            if cfg.release_control is not None and m["step"] == last_litho_step:
                released_pending_litho -= 1
                try_release(now)

    log = (pd.DataFrame(rows)
           .sort_values(["lot_id", "step_seq"])
           .reset_index(drop=True))

    lifecycle = pd.DataFrame({
        "lot_id": list(arrivals.keys()),
        "arrival_time": list(arrivals.values()),
    })
    lifecycle["completion_time"] = lifecycle["lot_id"].map(completions)
    lifecycle["due_date"] = lifecycle["lot_id"].map(due_dates)   # M9, append-only

    rho = theoretical_utilization(cfg)
    bottleneck = max(rho, key=rho.get)
    meta = {
        "seed": cfg.seed,
        "arrival_rate": cfg.arrival_rate,
        "horizon_hours": cfg.horizon_hours,
        "warmup_hours": cfg.warmup_hours,
        "route": cfg.route,
        "stations": {
            s: {"n_tools": st.n_tools, "batch_size": st.batch_size,
                "pt_mean": st.pt_mean, "pt_cv": st.pt_cv}
            for s, st in cfg.stations.items()
        },
        "theoretical_utilization": rho,
        "ground_truth_bottleneck": bottleneck,
        "anomalies": [a.label() for a in anomalies],
        "queue_discipline": cfg.queue_discipline,
        "flow_factor": cfg.flow_factor,
    }
    return log, lifecycle, meta


def default_config(seed: int = 42) -> FactoryConfig:
    """
    The locked configuration: a stylized single-layer wafer-fab loop.
      - 7 stations, single product; a lot represents a 25-wafer carrier (FOUP)
      - re-entrant route visiting LITHO twice (two mask layers)
      - FURNACE is a batch tool (2 tools x 4-lot batches, recipe-like low cv);
        slot utilization ~0.375 - deliberately NOT the constraint
      - LITHO engineered as the bottleneck (highest slot utilization, ~0.85),
        matching the real-fab pattern of scanners run closest to saturation
      - 60-day horizon (hours), 6-day warm-up

    Planned slot utilizations (arrival 1 lot/h):
      CLEAN 0.50, FURNACE 0.375, DEPO 0.65, LITHO 0.85 (bottleneck),
      ETCH 0.50, IMPLANT 0.55, METRO 0.45.
    """
    stations = {
        "CLEAN":   StationConfig("CLEAN",   pt_mean=1.0,  n_tools=2),
        "FURNACE": StationConfig("FURNACE", pt_mean=3.0,  n_tools=2,
                                 pt_cv=0.3, batch_size=4),   # batch tool
        "DEPO":    StationConfig("DEPO",    pt_mean=1.3,  n_tools=2),
        "LITHO":   StationConfig("LITHO",   pt_mean=0.85, n_tools=2),  # bottleneck
        "ETCH":    StationConfig("ETCH",    pt_mean=1.0,  n_tools=2),
        "IMPLANT": StationConfig("IMPLANT", pt_mean=1.1,  n_tools=2),
        "METRO":   StationConfig("METRO",   pt_mean=0.9,  n_tools=2),
    }
    # Stylized layer loop; LITHO re-entrant (two mask layers).
    route = ["CLEAN", "FURNACE", "DEPO", "LITHO", "ETCH", "LITHO",
             "IMPLANT", "METRO"]
    return FactoryConfig(
        stations=stations,
        route=route,
        arrival_rate=1.0,        # 1 lot/hour
        horizon_hours=60 * 24,   # 60 days
        warmup_hours=6 * 24,     # 6 days
        seed=seed,
    )
