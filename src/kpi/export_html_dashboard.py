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
            text="Manufacturing Ops Analytics: KPI Baseline "
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
<title>KPI Baseline: Manufacturing Operations Analytics</title>
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
  .howto b {{ color:#1565C0; }}
  .finding-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin:1rem 0 1rem; }}
  .finding-card {{ background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.85rem .95rem; }}
  .finding-card h3 {{ font-size:.92rem; margin:.05rem 0 .35rem; line-height:1.35; }}
  .finding-card p {{ margin:.3rem 0 0; color:#5B7180; font-size:.84rem; line-height:1.5; }}
  .tags {{ display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.55rem; }}
  .tag {{ display:inline-block; background:#E3F0FB; color:#1565C0; border-radius:999px;
          padding:.12rem .45rem; font-size:.68rem; font-weight:700; }}
  .chart-roles {{ display:grid; grid-template-columns:repeat(5,1fr); gap:.55rem; margin:.8rem 0 1.2rem; }}
  .role {{ background:#fff; border:1px solid #DCE3E8; border-radius:10px; padding:.65rem .7rem; }}
  .role b {{ color:#1565C0; font-size:.82rem; }}
  .role p {{ margin:.25rem 0 0; color:#5B7180; font-size:.78rem; line-height:1.45; }}
  .plot {{ background:#fff; border:1px solid #DCE3E8; border-radius:12px; padding:.4rem; }}
  .note {{ font-size:.8rem; color:#9AA9B4; margin-top:1.2rem; }}
  @media (max-width:720px) {{
    .topbar {{ align-items:flex-start; gap:.75rem; }}
    .finding-grid, .chart-roles {{ grid-template-columns:1fr; }}
  }}
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
  <div class="en"><h1>KPI Baseline: The starting point for the three findings</h1>
    <p class="lede">These charts are not a standalone dashboard. They are the shared baseline
    behind the three findings. Output, WIP, utilization, cycle time, and X-factor provide
    the reference point for bottleneck diagnosis, investment trade-off, and degradation monitoring.</p></div>
  <div class="zh"><h1>KPI Baseline：三個 findings 的共同起點</h1>
    <p class="lede">這些圖不是單獨展示的 dashboard，而是三個 findings 共用的基準證據。Output、WIP、utilization、cycle time 和 X-factor 提供判斷瓶頸、比較投資取捨，以及監控劣化是否先造成等待成本的參考點。</p></div>

  <div class="howto">
    <h2><span class="en">How this baseline supports the findings</span><span class="zh">這組 baseline 如何支援三個 findings</span></h2>
    <p class="en">The baseline charts identify signals and context. The findings then test those signals with what-if simulation, cost modeling, and monitoring backtests.</p>
    <p class="zh">Baseline 圖表提供訊號與背景，後面的 findings 再用 what-if simulation、成本模型與 monitoring backtest 進一步測試。</p>
  </div>

  <div class="finding-grid">
    <div class="finding-card">
      <div class="en"><h3>Finding 01: Bottleneck diagnosis starts from local signals</h3>
        <p>Utilization shows which stations are close to full load. WIP and cycle time show whether lots are waiting and accumulating. These signals identify bottleneck candidates, then Finding 01 uses +1 tool what-if tests to verify whether LITHO is the true constraint.</p></div>
      <div class="zh"><h3>Finding 01：瓶頸診斷從局部訊號開始</h3>
        <p>Utilization 顯示哪些站點接近滿載。WIP 和 cycle time 顯示 lot 是否正在等待與累積。這些訊號指出瓶頸候選站點，Finding 01 再用 +1 tool what-if 測試確認 LITHO 是否是真正限制。</p></div>
      <div class="tags"><span class="tag">Utilization</span><span class="tag">WIP</span><span class="tag">Cycle Time</span></div>
    </div>
    <div class="finding-card">
      <div class="en"><h3>Finding 02: Investment decisions need more than cycle-time reduction</h3>
        <p>Cycle time measures the operational impact of an improvement, but capacity decisions cannot rely only on time reduction. Finding 02 connects +1 tool scenarios to cost impact and checks whether cycle-time improvement and investment cost create a trade-off.</p></div>
      <div class="zh"><h3>Finding 02：投資決策不能只看 cycle time 降幅</h3>
        <p>Cycle time 衡量改善方案的營運影響，但產能決策不能只看時間下降。Finding 02 把 +1 tool 情境連到成本影響，檢查 cycle-time 改善與投資成本是否形成 trade-off。</p></div>
      <div class="tags"><span class="tag">Cycle Time</span><span class="tag">Output</span><span class="tag">What-if Cost</span></div>
    </div>
    <div class="finding-card">
      <div class="en"><h3>Finding 03: Degradation can appear before output drops</h3>
        <p>Output shows whether production has visibly dropped. Cycle time, WIP, and X-factor show whether waiting and congestion are already building up. Finding 03 compares output-only monitoring with drift monitoring before output makes the issue obvious.</p></div>
      <div class="zh"><h3>Finding 03：劣化可能先出現在 cycle time，而不是 output</h3>
        <p>Output 顯示產出是否明顯下降。Cycle time、WIP 和 X-factor 顯示等待與壅塞是否已經累積。Finding 03 用這個想法比較只看 output 的監控與 drift monitoring。</p></div>
      <div class="tags"><span class="tag">Output</span><span class="tag">Cycle Time</span><span class="tag">WIP</span><span class="tag">X-factor</span></div>
    </div>
  </div>

  <div class="chart-roles">
    <div class="role"><b>Output</b><p class="en">Comparison signal for Finding 03. If output stays normal, slow degradation can remain hidden.</p><p class="zh">Finding 03 的比較訊號。若 output 仍接近正常，慢性劣化可能被隱藏。</p></div>
    <div class="role"><b>WIP</b><p class="en">Shows whether lots are accumulating in the system before output makes it obvious.</p><p class="zh">顯示 lot 是否已在系統中累積，通常會早於 output 明顯變差。</p></div>
    <div class="role"><b>Utilization</b><p class="en">Identifies bottleneck candidates, which Finding 01 then tests with system-level what-if simulation.</p><p class="zh">指出瓶頸候選站點，Finding 01 再用系統層級 what-if simulation 測試。</p></div>
    <div class="role"><b>Cycle Time</b><p class="en">Core metric across all three findings: bottleneck impact, investment comparison, and drift monitoring.</p><p class="zh">三個 findings 的核心指標：瓶頸影響、投資比較與劣化監控都會用到。</p></div>
    <div class="role"><b>X-factor</b><p class="en">Shows how much actual cycle time is inflated by waiting and congestion, useful for hidden pressure.</p><p class="zh">顯示實際 cycle time 被等待與壅塞放大多少，可用來觀察隱藏壓力。</p></div>
  </div>

  <div class="plot">{plot_div}</div>

  <p class="note en">Synthetic data, clearly labeled. Cost/rate assumptions live in the notebooks;
  this page shows physical KPIs only. The baseline charts suggest signals, while the findings
  test those signals with additional analysis.</p>
  <p class="note zh">合成資料，明確標示。成本與費率假設放在 notebook，本頁只呈現物理 KPI。Baseline 圖表提供訊號，findings 會用後續分析測試這些訊號。</p>
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
