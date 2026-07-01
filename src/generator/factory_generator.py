"""
Synthetic fab-style production-line generator (Milestone M2).

A transparent, hand-written discrete-event simulation (DES) of an open
multi-server queueing network. No external DES library is used so that every
modeling step is explicit and defensible.

Design choices (each is a deliberate, defendable trade-off):

- Open queueing network with FIFO multi-server stations. FIFO is the simplest
  baseline dispatch rule. Real fabs use far more complex dispatching; that is a
  documented limitation / future work, not an oversight.

- Re-entrant route: a station can appear multiple times in the route. This is a
  defining feature of semiconductor fabs (e.g., litho/etch revisited) and is the
  structural idea borrowed from the public SMT2020 testbed. It is also what makes
  the designed bottleneck loaded.

- Lognormal processing times: positive and right-skewed, which matches real
  process-time behavior. The coefficient of variation (cv) controls variability;
  variability is what creates queueing (Kingman intuition), so WIP builds in
  front of the busiest station.

- Poisson arrivals (exponential inter-arrival times): the standard memoryless
  arrival assumption for an open line.

- A single station is engineered to have the least capacity headroom (highest
  utilization). By Theory of Constraints it sets line throughput, so it is the
  ground-truth bottleneck recorded in metadata and later used to validate the
  M4 detector.

The simulation is fully seeded for reproducibility.

Common Random Numbers (CRN) — M4 addition
------------------------------------------
For the M4 counterfactual we need a *paired* comparison: baseline vs "+1 tool at
station X" must face the SAME random inputs so the measured delta reflects only
the capacity change, not a different random stream. To support that, all
randomness can be pre-drawn into a ``RandomDraws`` table via ``draw_randoms()``
and passed to ``simulate(cfg, draws=...)``; the event loop then consumes the
table and calls no RNG at all.

- ``simulate(cfg)`` with ``draws=None`` is the ORIGINAL M2 code path, byte-for-byte
  unchanged: it samples lazily inside the event loop from an internal RNG seeded
  by ``cfg.seed``. M2/M3 artifacts are unaffected by the CRN refactor.
- ``simulate(cfg, draws)`` with an explicit table is fully deterministic (no RNG),
  which is what makes baseline-vs-baseline on the same table produce an exact zero
  delta — the sanity check that proves no hidden RNG source escapes the table.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StationConfig:
    """Configuration for one station (tool group)."""
    name: str
    pt_mean: float          # mean processing time per operation (hours)
    n_tools: int            # number of parallel tools (servers)
    pt_cv: float = 0.5      # coefficient of variation of processing time


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

        IMPORTANT — indexing is by ROUTE STEP (visit order), NOT by station.
        The route is re-entrant: ``["S1","S2","S3","S4","S5","S4","S6","S7"]``
        visits S4 twice, at step 3 and step 5. Those are two INDEPENDENT draws,
        ``proc_times[lot][3]`` and ``proc_times[lot][5]``. Because the pairing is
        by step, baseline and any "+1 tool" treatment consume the exact same two
        S4 draws in the exact same order — a re-entrant station cannot get its
        paired draws mis-aligned. Every lot traverses the full route exactly once
        (no rework in this model), so ``len(proc_times[lot]) == len(route)`` and
        the table is consumed identically regardless of ``n_tools``.
    """
    arrivals: list           # arrivals[lot_id] -> arrival time
    proc_times: list         # proc_times[lot_id][step] -> processing hours


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert a target mean and CV into lognormal (mu, sigma) parameters."""
    sigma2 = math.log(1.0 + cv ** 2)
    mu = math.log(mean) - sigma2 / 2.0
    return mu, math.sqrt(sigma2)


def theoretical_utilization(cfg: FactoryConfig) -> dict:
    """
    Design-time utilization per station:
        rho_s = arrival_rate * visits_s * pt_mean_s / n_tools_s

    This is the planned load and identifies the intended bottleneck before any
    simulation runs. The DES should reproduce this ordering empirically.
    """
    visits = {s: cfg.route.count(s) for s in cfg.stations}
    rho = {}
    for s, st in cfg.stations.items():
        rho[s] = cfg.arrival_rate * visits[s] * st.pt_mean / st.n_tools
    return rho


def draw_randoms(cfg: FactoryConfig, seed: int) -> RandomDraws:
    """Pre-draw all randomness for one replication into a reusable table (CRN).

    The returned table depends only on ``seed`` and the *distributional* config
    (arrival_rate, route, and each station's pt_mean / pt_cv). It does NOT depend
    on ``n_tools``. That is the whole point: generate one table per replication,
    then run the baseline and every "+1 tool" scenario against that SAME table so
    they face identical arrivals and identical per-visit processing times, and the
    only thing that varies is capacity.

    Draw order (single RNG stream, documented for reproducibility):
      1. Inter-arrival times, accumulated until ``horizon_hours`` (same rule the
         legacy ``draws=None`` path uses to schedule arrivals).
      2. Then, per lot in arrival order, one processing-time draw per route step,
         in step order. See ``RandomDraws.proc_times`` for the by-step indexing
         and why it keeps re-entrant S4 paired correctly.
    """
    rng = np.random.default_rng(seed)

    # 1) Arrivals — identical generation rule to the legacy path.
    arrivals: list = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / cfg.arrival_rate)
        if t >= cfg.horizon_hours:
            break
        arrivals.append(t)

    # 2) Processing times, indexed by (lot, route step). S4 at steps 3 and 5 gets
    #    two independent draws here; both are reused by baseline and treatment.
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


def simulate(cfg: FactoryConfig, draws: RandomDraws | None = None):
    """
    Run the discrete-event simulation.

    Parameters
    ----------
    cfg : FactoryConfig
        Factory / experiment configuration.
    draws : RandomDraws | None
        If ``None`` (default), randomness is sampled lazily inside the event loop
        from an internal RNG seeded by ``cfg.seed`` — this is the ORIGINAL M2
        behaviour, kept byte-for-byte identical. If a ``RandomDraws`` table is
        provided (Common Random Numbers), the loop consumes it and calls NO RNG,
        so the run is fully deterministic and paired against any other run that
        uses the same table.

    Returns
    -------
    log : pd.DataFrame
        One row per completed operation:
        [lot_id, product_type, step_seq, station,
         queue_entry_time, process_start_time, process_complete_time]
    lifecycle : pd.DataFrame
        One row per lot: [lot_id, arrival_time, completion_time].
    meta : dict
        Configuration echo + ground-truth bottleneck.
    """
    # RNG exists ONLY on the legacy (draws=None) path. On the CRN path it stays
    # None and must never be touched — if it were, baseline-vs-baseline on one
    # table would not be an exact zero and the CRN sanity check would catch it.
    rng = np.random.default_rng(cfg.seed) if draws is None else None

    free = {s: st.n_tools for s, st in cfg.stations.items()}   # free tools per station
    pending = {s: [] for s in cfg.stations}                    # FIFO queues
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
        """Processing time for ``lot`` at route position ``step`` (station ``s``).

        Legacy path (draws=None): draw lazily, preserving the original RNG call
        order exactly. CRN path: read the pre-drawn value indexed by (lot, step)
        — by step, so re-entrant S4 (steps 3 and 5) stays paired.
        """
        if draws is None:
            return sample_pt(s)
        return draws.proc_times[lot][step]

    def request(lot, step, now):
        """Lot requests the station for this route step."""
        s = cfg.route[step]
        if free[s] > 0:
            free[s] -= 1
            pt = pt_for(lot, step, s)
            push(now + pt, "complete",
                 {"lot": lot, "step": step, "qentry": now, "start": now})
        else:
            pending[s].append({"lot": lot, "step": step, "qentry": now})

    # Schedule arrivals up front.
    if draws is None:
        # Legacy: Poisson arrivals sampled from the internal RNG (unchanged).
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

        # kind == "complete"
        s = cfg.route[p["step"]]
        rows.append({
            "lot_id": p["lot"],
            "product_type": cfg.product_type,
            "step_seq": p["step"],
            "station": s,
            "queue_entry_time": p["qentry"],
            "process_start_time": p["start"],
            "process_complete_time": now,
        })
        free[s] += 1

        # A tool just freed: pull the next waiting lot at this station (FIFO).
        if pending[s]:
            nxt = pending[s].pop(0)
            free[s] -= 1
            pt = pt_for(nxt["lot"], nxt["step"], s)
            push(now + pt, "complete",
                 {"lot": nxt["lot"], "step": nxt["step"],
                  "qentry": nxt["qentry"], "start": now})

        # Advance the completed lot to its next route step (or finish).
        nstep = p["step"] + 1
        if nstep < len(cfg.route):
            request(p["lot"], nstep, now)
        else:
            completions[p["lot"]] = now

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
        "theoretical_utilization": rho,
        "ground_truth_bottleneck": bottleneck,
    }
    return log, lifecycle, meta


def default_config(seed: int = 42) -> FactoryConfig:
    """
    The agreed M2 starting configuration:
      - 7 stations (S1..S7), single product
      - re-entrant route visiting S4 twice
      - S4 engineered as the bottleneck (highest planned utilization, ~0.85)
      - 60-day horizon (hours), 6-day warm-up
    """
    stations = {
        "S1": StationConfig("S1", pt_mean=1.0, n_tools=2),
        "S2": StationConfig("S2", pt_mean=1.2, n_tools=2),
        "S3": StationConfig("S3", pt_mean=1.3, n_tools=2),
        "S4": StationConfig("S4", pt_mean=0.85, n_tools=2),   # bottleneck
        "S5": StationConfig("S5", pt_mean=1.0, n_tools=2),
        "S6": StationConfig("S6", pt_mean=1.1, n_tools=2),
        "S7": StationConfig("S7", pt_mean=0.9, n_tools=2),
    }
    route = ["S1", "S2", "S3", "S4", "S5", "S4", "S6", "S7"]  # S4 re-entrant
    return FactoryConfig(
        stations=stations,
        route=route,
        arrival_rate=1.0,        # 1 lot/hour
        horizon_hours=60 * 24,   # 60 days
        warmup_hours=6 * 24,     # 6 days
        seed=seed,
    )
