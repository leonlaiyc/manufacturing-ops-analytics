"""
Synthetic fab-style production-line generator (Milestone M2, fab-ized in M7).

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
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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
class FactoryConfig:
    """Full factory / experiment configuration."""
    stations: dict          # name -> StationConfig
    route: list             # ordered station names; repeats = re-entrant flow
    arrival_rate: float     # lots per hour (Poisson process)
    horizon_hours: float    # total simulated time
    warmup_hours: float     # initial period excluded from steady-state stats
    seed: int = 42
    product_type: str = "P1"


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
    waiting lots as ONE run on one tool (greedy load-and-go, FIFO order). The
    run's processing time is the first-loaded lot's draw; all members share the
    same process_start / process_complete and then advance individually.

    Returns
    -------
    log : pd.DataFrame
        One row per completed operation (batch members get one row each):
        [lot_id, product_type, step_seq, station,
         queue_entry_time, process_start_time, process_complete_time, tool_id]
    lifecycle : pd.DataFrame
        One row per lot: [lot_id, arrival_time, completion_time].
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
    pending = {s: [] for s in cfg.stations}                    # FIFO queues
    tools = {s: _ToolPool(s, st.n_tools) for s, st in cfg.stations.items()}  # M7
    rows = []
    arrivals: dict[int, float] = {}
    completions: dict[int, float] = {}

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
        and 5) stays paired.
        """
        if draws is None:
            return sample_pt(s)
        return draws.proc_times[lot][step]

    def try_dispatch(s, now):
        """Start as many runs as free tools allow; each takes up to batch_size."""
        B = cfg.stations[s].batch_size
        offsets = cfg.stations[s].tool_offsets
        while free[s] > 0 and pending[s]:
            k = min(B, len(pending[s]))
            members = [pending[s].pop(0) for _ in range(k)]
            free[s] -= 1
            tool_idx = tools[s].acquire()
            lead = members[0]
            pt = pt_for(lead["lot"], lead["step"], s)
            if offsets is not None:
                pt *= offsets[tool_idx]          # M7: post-draw scale only, no RNG
            push(now + pt, "complete",
                 {"members": members, "start": now, "tool_idx": tool_idx})

    def request(lot, step, now):
        """Lot requests the station for this route step (FIFO queue + dispatch)."""
        s = cfg.route[step]
        pending[s].append({"lot": lot, "step": step, "qentry": now})
        try_dispatch(s, now)

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
            arrivals[p["lot"]] = now
            request(p["lot"], 0, now)
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

        # A tool just freed: pull the next waiting run at this station (FIFO).
        try_dispatch(s, now)

        # Advance each completed lot to its next route step (or finish).
        for m in members:
            nstep = m["step"] + 1
            if nstep < len(cfg.route):
                request(m["lot"], nstep, now)
            else:
                completions[m["lot"]] = now

    log = (pd.DataFrame(rows)
           .sort_values(["lot_id", "step_seq"])
           .reset_index(drop=True))

    lifecycle = pd.DataFrame({
        "lot_id": list(arrivals.keys()),
        "arrival_time": list(arrivals.values()),
    })
    lifecycle["completion_time"] = lifecycle["lot_id"].map(completions)

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
    """
    rng = np.random.default_rng(cfg.seed) if draws is None else None

    base_tools = {s: st.n_tools for s, st in cfg.stations.items()}
    busy = {s: 0 for s in cfg.stations}                        # tools in service
    pending = {s: [] for s in cfg.stations}                    # FIFO queues
    # M7: pool sized to the station's full (nominal) tool count. A breakdown only
    # limits how many of these indices may be acquired concurrently (via
    # effective_capacity below); it does not change which indices exist, so
    # tool_id labeling and offsets stay well-defined even during a breakdown window.
    tools = {s: _ToolPool(s, st.n_tools) for s, st in cfg.stations.items()}
    rows = []
    arrivals: dict[int, float] = {}
    completions: dict[int, float] = {}
    extra_pts: dict[int, list] = {}                            # surge lot -> proc times

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
        # Start as many runs as the (possibly reduced) capacity allows, FIFO;
        # each run takes up to batch_size waiting lots.
        B = cfg.stations[s].batch_size
        offsets = cfg.stations[s].tool_offsets
        while pending[s] and busy[s] < effective_capacity(s, now):
            k = min(B, len(pending[s]))
            members = [pending[s].pop(0) for _ in range(k)]
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
        pending[s].append({"lot": lot, "step": step, "qentry": now})
        try_dispatch(s, now)

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
            arrivals[p["lot"]] = now
            request(p["lot"], 0, now)
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
        try_dispatch(s, now)                 # pull next waiting run (FIFO)

        for m in members:
            nstep = m["step"] + 1
            if nstep < len(cfg.route):
                request(m["lot"], nstep, now)
            else:
                completions[m["lot"]] = now

    log = (pd.DataFrame(rows)
           .sort_values(["lot_id", "step_seq"])
           .reset_index(drop=True))

    lifecycle = pd.DataFrame({
        "lot_id": list(arrivals.keys()),
        "arrival_time": list(arrivals.values()),
    })
    lifecycle["completion_time"] = lifecycle["lot_id"].map(completions)

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
