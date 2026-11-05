"""Phase 3: network forecast (docs/phase3.md).

Baseline: the per-camera TrafficForecaster (gbm.py, this project's own copy of v1's design; v1 itself has
been cleaned up). The network model is the same model with neighbour features added:

  up|<edge>    flow of each edge ending at this edge's start junction (U-turns left out): now,
               5 and 10 minutes ago, 15-minute mean
  down|<edge>  load ratio at the stop line of each edge starting at this edge's end junction: now,
               15-minute mean
  jout         flow of all camera edges leaving this edge's start junction: now, 15-minute mean
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .gbm import ForecasterSettings, TrafficForecaster, persistence, seasonal_naive

TARGETS = ["flow", "pcu"]
HORIZONS = [3, 6, 12]                       # 15, 30, 60 minutes


def settings() -> ForecasterSettings:
    """v1's defaults (config.yaml forecast section), without the prediction interval."""
    return ForecasterSettings(targets=TARGETS, horizons=HORIZONS, quantiles=[])


def neighbours(edges: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    e = edges[edges.camera_id.notna()].set_index("edge_id")
    out = {}
    for x in e.index:
        u, v = e.at[x, "from_node"], e.at[x, "to_node"]
        out[x] = {"up": [a for a in e.index if e.at[a, "to_node"] == u and e.at[a, "from_node"] != v],
                  "down": [b for b in e.index if e.at[b, "from_node"] == v and e.at[b, "to_node"] != u],
                  "junction": [] if u.startswith("F_") else [b for b in e.index if e.at[b, "from_node"] == u]}
    return out


def wide(obs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """value -> slots x edges, for flow, pcu and load_ratio."""
    out = {}
    for value in ("flow", "pcu", "load_ratio"):
        w = obs.pivot(index="ts", columns="edge_id", values=value).astype(float)
        w.index = pd.to_datetime(w.index)
        out[value] = w.sort_index()
    return out


def camera_series(edge: str, w: dict[str, pd.DataFrame], nb: dict[str, list[str]]) -> pd.DataFrame:
    """One camera's series as v1 expects it (targets + observed), plus its neighbours' columns."""
    s = pd.DataFrame({t: w[t][edge] for t in TARGETS})
    s["observed"] = 1
    for a in nb["up"]:
        s[f"up|{a}"] = w["flow"][a]
    for b in nb["down"]:
        s[f"down|{b}"] = w["load_ratio"][b]
    if nb["junction"]:
        s["jout"] = w["flow"][nb["junction"]].sum(axis=1)
    return s


class NetworkForecaster(TrafficForecaster):
    """v1's forecaster with the neighbours' recent values as extra features."""

    backend = "network"

    def _features(self, series, target, h):
        X = super()._features(series, target, h)
        extra = {}
        for col in series.columns:
            x = series[col]
            if col.startswith("up|"):
                extra.update({f"{col}_now": x, f"{col}_lag1": x.shift(1), f"{col}_lag2": x.shift(2),
                              f"{col}_mean3": x.rolling(3, min_periods=1).mean()})
            elif col.startswith("down|") or col == "jout":
                extra.update({f"{col}_now": x, f"{col}_mean3": x.rolling(3, min_periods=1).mean()})
        return pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1)


def fit_camera(edge: str, series: pd.DataFrame, train_until: pd.Timestamp | None,
               hide: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> dict:
    """Both models for one camera. Training uses origins up to `train_until` (v1's hold-out) or, with
    `hide`, all days except that range (its slots marked unobserved so no target there is learned)."""
    s = series.copy()
    if hide is not None:
        s.loc[(s.index > hide[0]) & (s.index <= hide[1]), "observed"] = 0
    if train_until is not None:
        s = s.loc[:train_until]
    own = s[TARGETS + ["observed"]]
    return {"edge": edge, "v1": TrafficForecaster(settings(), edge).fit(own),
            "network": NetworkForecaster(settings(), edge).fit(s)}


def fit_all(series: dict[str, pd.DataFrame], workers: int = 4, **kw) -> dict[str, dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_camera, e, s, **kw) for e, s in series.items()]
        return {f.result()["edge"]: f.result() for f in futures}


def predictions(models: dict, series: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
    """Forecast for t + h made at every origin t, re-indexed to the target time."""
    own = series[TARGETS + ["observed"]]
    p = pd.DataFrame({"v1": models["v1"].predict_frame(own, target, h)["point"],
                      "network": models["network"].predict_frame(series, target, h)["point"],
                      "persistence": persistence(series, target, h),
                      "same_time_yesterday": seasonal_naive(series, target, h)}, index=series.index)
    p.index = p.index + pd.Timedelta(minutes=5 * h)
    p["actual"] = series[target].reindex(p.index).to_numpy()
    return p


def scores(p: pd.DataFrame) -> list[dict]:
    p = p.dropna(subset=["actual"])
    rows = []
    for m in ("v1", "network", "persistence", "same_time_yesterday"):
        ok = p[m].notna()
        err = (p[m][ok] - p.actual[ok]).abs()
        rows.append({"model": m, "n": int(ok.sum()), "mae": float(err.mean()),
                     "abs_err": float(err.sum()), "actual_sum": float(p.actual[ok].abs().sum())})
    return rows


def weeks(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(after, up to) of each 7-day block of the run."""
    start = index.min().normalize()
    n = int(np.ceil((index.max() - start) / pd.Timedelta(days=7)))
    return [(start + pd.Timedelta(days=7 * i), start + pd.Timedelta(days=7 * (i + 1))) for i in range(n)]
