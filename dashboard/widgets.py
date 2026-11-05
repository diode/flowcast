"""Widgets both overview views use: ECharts charts and HTML tiles."""
from __future__ import annotations

import math

import pandas as pd
from streamlit_echarts import JsCode, st_echarts

from netstudy import overview as ov


def js(option: dict) -> dict:
    """Turn the JS function strings in an option (formatters, label intervals) into functions for ECharts."""
    def walk(x):
        if isinstance(x, dict):
            return {k: (JsCode(v).js_code if isinstance(v, str) and v.startswith("function") else walk(v))
                    for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(option)


def chart(option: dict, height: int, key: str) -> None:
    st_echarts(js(option), height=f"{height}px", key=key)


def spark_svg(values: list[float], colour: str = ov.ACCENT, w: int = 120, h: int = 34) -> str:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(i * w / (len(v) - 1), h - 3 - (x - lo) / span * (h - 6)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + line + f" {w},{h}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polygon points="{area}" fill="{colour}" fill-opacity="0.10"/>'
            f'<polyline points="{line}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" '
            f'stroke-linecap="round"/><circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3" fill="{colour}"/></svg>')


def tile(label: str, value: str, unit: str, now: float, usual: float, spark: list, worse_when_up: bool,
         note: str) -> str:
    if pd.notna(now) and pd.notna(usual) and abs(usual) > 1e-9:
        pct = 100 * (now - usual) / abs(usual)
        up = pct > 0
        bad = up == worse_when_up and abs(pct) >= 5
        good = up != worse_when_up and abs(pct) >= 5
        cls = "bad" if bad else "good" if good else "flat"
        delta = f'<span class="delta {cls}">{"▲" if up else "▼"} {abs(pct):.0f}%</span> vs usual'
    elif pd.notna(now) and pd.notna(usual):
        delta = '<span class="delta flat">≈</span> as usual'
    else:
        delta = ""
    return (f'<div class="tile"><div class="tile-label">{label}</div>'
            f'<div class="tile-row"><div><span class="tile-value">{value}</span><span class="tile-unit">{unit}</span>'
            f'<div class="tile-delta">{delta}</div></div>{spark_svg(spark)}</div>'
            f'<div class="tile-note">{note}</div></div>')
