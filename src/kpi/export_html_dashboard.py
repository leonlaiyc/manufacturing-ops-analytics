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


# --------------------------------------------------------------------------- #
# Bilingual page shell (EN / 繁中) so a first-time viewer knows how to read the
# charts. The plot itself is language-neutral; only the surrounding copy toggles.
# --------------------------------------------------------------------------- #
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KPI Dashboard — Manufacturing Operations Analytics</title>
<style>
  body {{ font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans TC",Helvetica,Arial,sans-serif;
         color:#22303C; background:#F7F9FA; margin:0; line-height:1.6; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:0 1.1rem 3rem; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:center;
             max-width:1000px; margin:0 auto; padding:.9rem 1.1rem .2rem; }}
  .back {{ font-size:.86rem; color:#1565C0; text-decoration:none; font-weight:600; }}
  .langbtn {{ border:1px solid #DCE3E8; background:#fff; color:#5B7180; font-size:.8rem;
              padding:.28rem .66rem; border-radius:999px; cursor:pointer; font-weight:600; margin-left:.35rem; }}
  .langbtn[aria-pressed="true"] {{ background:#1565C0; color:#fff; border-color:#1565C0; }}
  .lang-zh .en {{ display:none; }} .lang-en .zh {{ display:none; }}
  h1 {{ font-size:1.35rem; margin:.7rem 0 .2rem; }}
  .lede {{ color:#5B7180; margin:.1rem 0 1rem; font-size:.98rem; }}
  .howto {{ background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.9rem 1.1rem;
            margin:0 0 1.2rem; }}
  .howto h2 {{ font-size:.82rem; letter-spacing:.05em; text-transform:uppercase; color:#5B7180;
               margin:0 0 .5rem; }}
  .howto ul {{ margin:0; padding-left:1.1rem; }}
  .howto li {{ font-size:.9rem; margin:.28rem 0; color:#22303C; }}
  .howto b {{ color:#1565C0; }}
  .plot {{ background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.4rem; }}
  .note {{ font-size:.8rem; color:#9AA9B4; margin-top:1.2rem; }}
</style>
</head>
<body class="lang-zh">
<div class="topbar">
  <a class="back" href="index.html"><span class="en">← Back to the 1-page report</span><span class="zh">← 回到一頁式報告</span></a>
  <div>
    <button class="langbtn" data-lang="zh" onclick="setLang('zh')">中文</button>
    <button class="langbtn" data-lang="en" onclick="setLang('en')">EN</button>
  </div>
</div>
<div class="wrap">
  <div class="en"><h1>KPI Dashboard — the fab-style line at a glance</h1>
    <p class="lede">The detailed data behind the report. Synthetic line, steady-state window only.
    Hover for values; drag to zoom.</p></div>
  <div class="zh"><h1>KPI 儀表板——一眼看懂這條 fab 式產線</h1>
    <p class="lede">報告背後的詳細數據。合成產線,僅取穩態區間。滑鼠移上去看數值、拖曳可放大。</p></div>

  <div class="howto">
    <h2><span class="en">How to read the six charts</span><span class="zh">六張圖怎麼看</span></h2>
    <ul class="en">
      <li><b>Daily output</b> — lots finished per day; the line's steady output rate (orange line = design target).</li>
      <li><b>WIP over time</b> — how much work is on the line at once; the shaded start is warm-up (excluded).</li>
      <li><b>Slot utilization</b> — how loaded each station is; <b>LITHO (red) is the bottleneck</b> at ~85%, diamonds = design target.</li>
      <li><b>Daily median cycle time</b> — how long a lot takes, day by day; flat = healthy and stable.</li>
      <li><b>Cycle-time distribution</b> — the spread across lots; the right tail is the unlucky ones.</li>
      <li><b>X-factor</b> — cycle time ÷ pure processing time; the fab's headline flow metric. 1 = zero waiting, higher = more queueing.</li>
    </ul>
    <ul class="zh">
      <li><b>每日產出</b>——每天完成幾個 lot,產線的穩定產出率(橘線=設計目標)。</li>
      <li><b>在製品(WIP)</b>——產線上同時有多少在製品;開頭陰影是暖機期(不列入統計)。</li>
      <li><b>Slot 稼動率</b>——每一站有多滿;<b>LITHO(紅)就是瓶頸</b>,約 85%,菱形=設計目標。</li>
      <li><b>每日中位生產週期</b>——一個 lot 要多久,逐日看;平穩=健康穩定。</li>
      <li><b>生產週期分布</b>——各 lot 的落點分布;右邊長尾是比較倒楣的那些。</li>
      <li><b>X-factor</b>——生產週期 ÷ 純加工時間,fab 的頭號流動指標。1=完全沒排隊,越高=排隊越久。</li>
    </ul>
  </div>

  <div class="plot">{plot_div}</div>

  <p class="note en">Synthetic data, clearly labeled. Cost/rate assumptions live in the notebooks;
  this page shows physical KPIs only.</p>
  <p class="note zh">合成資料,明確標示。成本/費率假設放在 notebook;本頁只呈現物理 KPI。</p>
</div>
<script>
  function setLang(l){{
    document.body.className = 'lang-' + l;
    document.documentElement.lang = (l === 'zh') ? 'zh-Hant' : 'en';
    try {{ localStorage.setItem('mfg_lang', l); }} catch(e){{}}
    document.querySelectorAll('.langbtn').forEach(function(b){{
      b.setAttribute('aria-pressed', b.dataset.lang === l);
    }});
  }}
  var saved; try {{ saved = localStorage.getItem('mfg_lang'); }} catch(e){{}}
  setLang(saved || 'zh');
</script>
</body>
</html>
"""


def main() -> None:
    fig = build_figure()
    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False)
    html = PAGE_TEMPLATE.format(plot_div=plot_div)
    for out in (OUT_REPORT, OUT_DOCS):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
