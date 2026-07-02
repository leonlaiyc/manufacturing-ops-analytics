"""
Export the M3 KPI dashboard as a standalone interactive HTML page.

Reads the committed synthetic artifacts (data/synthetic/) and writes one
self-contained Plotly page (plotly.js via CDN, so the file stays small) to:

  - reports/html/03_kpi_dashboard.html   (repo artifact)
  - docs/dashboard.html                  (published via GitHub Pages)

The page mirrors the static notebook 03 charts — daily output, WIP, slot
utilization vs design targets, cycle time, X-factor — with hover/zoom, and
carries the same honest-scope labeling (synthetic data, clearly stated).

Run:  python src/kpi/export_html_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kpi.kpi_metrics import (          # noqa: E402
    cycle_time_stats,
    daily_median_ct,
    daily_throughput,
    station_utilization,
    wip_timeseries,
    x_factor,
)

DATA = ROOT / "data" / "synthetic"
OUT_REPORT = ROOT / "reports" / "html" / "03_kpi_dashboard.html"
OUT_DOCS = ROOT / "docs" / "dashboard.html"


def build_figure() -> go.Figure:
    event_log = pd.read_csv(DATA / "event_log.csv")
    lifecycle = pd.read_csv(DATA / "lot_lifecycle.csv")
    with open(DATA / "metadata.json") as f:
        meta = json.load(f)

    t0, t1 = meta["warmup_hours"], meta["horizon_hours"]
    stations = meta["stations"]
    theo = meta["theoretical_utilization"]

    thr = daily_throughput(lifecycle, t0, t1)
    times, wip = wip_timeseries(lifecycle)
    util = station_utilization(event_log, t0, t1, stations)
    ct_series, ct_med, ct_p90 = cycle_time_stats(lifecycle, t0, t1)
    dct = daily_median_ct(lifecycle, t0, t1)
    x_series, x_med, x_p90 = x_factor(event_log, lifecycle, t0, t1)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Daily output (lots completed)",
            "WIP over time (warm-up shaded)",
            "Slot utilization vs design target (LITHO = engineered bottleneck)",
            "Daily median cycle time (h)",
            "Cycle time distribution (h)",
            f"X-factor per lot (median {x_med:.2f}, p90 {x_p90:.2f})",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.08,
    )

    # 1 — daily throughput
    fig.add_trace(go.Bar(x=thr["day"], y=thr["count"], name="lots/day",
                         marker_color="#4878A8"), row=1, col=1)
    fig.add_hline(y=meta["validation"]["throughput_per_hour"] * 24,
                  line_dash="dash", line_color="darkorange", row=1, col=1)

    # 2 — WIP step function (thin the point cloud for page weight)
    step = max(1, len(times) // 4000)
    fig.add_trace(go.Scatter(x=times[::step], y=wip[::step], mode="lines",
                             name="WIP", line=dict(color="#4878A8", width=1)),
                  row=1, col=2)
    fig.add_vline(x=t0, line_dash="dash", line_color="crimson", row=1, col=2)

    # 3 — slot utilization
    colors = ["#EF5350" if s == meta["ground_truth_bottleneck"] else "#90A4AE"
              for s in util["station"]]
    fig.add_trace(go.Bar(x=util["station"], y=util["utilization"],
                         name="empirical", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=util["station"],
                             y=[theo[s] for s in util["station"]],
                             mode="markers", name="design target",
                             marker=dict(symbol="diamond", size=10,
                                         color="darkorange")), row=2, col=1)

    # 4 — daily median cycle time
    fig.add_trace(go.Scatter(x=dct["day"], y=dct["median_ct"], mode="lines+markers",
                             name="median CT", line=dict(color="#4878A8")),
                  row=2, col=2)

    # 5 — cycle time histogram
    fig.add_trace(go.Histogram(x=ct_series, nbinsx=40, name="cycle time",
                               marker_color="#4878A8"), row=3, col=1)
    fig.add_vline(x=ct_med, line_dash="dash", line_color="darkorange", row=3, col=1)
    fig.add_vline(x=ct_p90, line_dash="dot", line_color="crimson", row=3, col=1)

    # 6 — X-factor histogram
    fig.add_trace(go.Histogram(x=x_series, nbinsx=40, name="X-factor",
                               marker_color="#00897B"), row=3, col=2)
    fig.add_vline(x=x_med, line_dash="dash", line_color="darkorange", row=3, col=2)
    fig.add_vline(x=x_p90, line_dash="dot", line_color="crimson", row=3, col=2)

    fig.update_layout(
        title=dict(
            text="Manufacturing Ops Analytics — KPI Dashboard "
                 "<span style='font-size:13px;color:#888'>(SYNTHETIC fab-style "
                 "line: CLEAN → FURNACE(batch) → DEPO → LITHO → ETCH → LITHO → "
                 "IMPLANT → METRO; 1 lot = 25-wafer FOUP; steady-state window "
                 "only)</span>",
            x=0.5),
        height=1050, showlegend=False, template="plotly_white",
        margin=dict(t=90, l=60, r=40, b=50),
    )
    fig.update_yaxes(range=[0, 1.0], row=2, col=1)
    return fig


def main() -> None:
    fig = build_figure()
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    for out in (OUT_REPORT, OUT_DOCS):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
