"""Estimate edges whose camera is hidden (never had one), from the cameras that remain.

Flow (vehicles per 5 minutes):
  B0  mean flow of the visible road edges in the slot
  B1  flow of the other direction of the same road (if visible)
  M1  junction balance: at every junction vehicles in = vehicles out. Unknowns = hidden edges; solved
      per slot on 15-minute (3-slot) means by least squares. Where the equations leave a choice, the
      prior (B1, else B0) fills it in; negatives clipped to 0.
Load and level (what a camera at the stop line would see):
  B0 / B1 the same idea with load; M1 a gradient-boosting model trained on the visible road edges
  (features: flow, hour, day type, length, load of the visible approaches into the same junction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL_BINS = [-np.inf, 0.35, 0.65, 0.9, np.inf]          # v1 thresholds (simulate.DEFAULT_LEVELS)
LEVEL_NAMES = ["free", "moderate", "heavy", "jammed"]
CONGESTED = ("heavy", "jammed")
SMOOTH_SLOTS = 3                                          # 15 minutes


def level_of(load: pd.DataFrame | pd.Series):
    cut = lambda s: pd.cut(s, LEVEL_BINS, labels=LEVEL_NAMES).astype(str)  # noqa: E731
    return load.apply(cut) if isinstance(load, pd.DataFrame) else cut(load)


def junction_rows(edges: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Coefficients of 'in - out = 0' per junction: +1 edge ends at the junction, -1 edge starts there."""
    js = sorted(set(edges.to_node[~edges.to_node.str.startswith("F_")]))
    a = pd.DataFrame(0.0, index=js, columns=edges.edge_id)
    for e in edges.itertuples():
        if e.to_node in a.index:
            a.loc[e.to_node, e.edge_id] += 1
        if e.from_node in a.index:
            a.loc[e.from_node, e.edge_id] -= 1
    return js, a


def determined(a_hidden: np.ndarray) -> np.ndarray:
    """True for each hidden edge whose flow the junction equations fix exactly."""
    proj = np.linalg.pinv(a_hidden) @ a_hidden            # projector on the row space
    return np.isclose(proj, np.eye(len(proj)), atol=1e-8).all(axis=0)


def opposite(edge_id: str) -> str:
    u, v = edge_id.split("_")
    return f"{v}_{u}"


def estimate_flow(flow: pd.DataFrame, edges: pd.DataFrame, hidden: list[str]) -> dict[str, pd.DataFrame]:
    """flow: slots x edges (all cameras). Returns {method: slots x hidden edges}."""
    smooth_f = smooth(flow)
    roads = edges.edge_id[edges.kind == "road"]
    visible_roads = [e for e in roads if e not in hidden] or [e for e in flow.columns if e not in hidden]
    b0 = pd.DataFrame({h: smooth_f[visible_roads].mean(axis=1) for h in hidden})
    b1 = pd.DataFrame({h: smooth_f[opposite(h)] if opposite(h) not in hidden else np.nan for h in hidden},
                      index=smooth_f.index)
    prior = b1.fillna(b0)
    _, a = junction_rows(edges)
    known = [e for e in a.columns if e not in hidden]
    a_h, a_k = a[hidden].to_numpy(), a[known].to_numpy()
    b = -(smooth_f[known].to_numpy() @ a_k.T)               # A_h x = -A_k f_known, one row per slot
    pinv = np.linalg.pinv(a_h)
    free = np.eye(len(hidden)) - pinv @ a_h                 # directions the equations leave open
    x = b @ pinv.T + prior.to_numpy() @ free.T              # least squares + prior where undetermined
    m1 = pd.DataFrame(np.clip(x, 0, None), index=flow.index, columns=hidden)
    return {"B0": b0, "B1": b1, "M1": m1}


def load_features(edge_ids: list[str], flow: pd.DataFrame, load: pd.DataFrame, edges: pd.DataFrame,
                  hidden: list[str]) -> pd.DataFrame:
    """One row per (slot, edge): the model's features, from visible cameras only (and `flow` given)."""
    e = edges.set_index("edge_id")
    start = pd.to_datetime(flow.index) - pd.Timedelta(minutes=5)
    parts = []
    for eid in edge_ids:
        j = e.at[eid, "to_node"]
        others = [x for x in e.index[e.to_node == j] if x != eid and x not in hidden]
        parts.append(pd.DataFrame({
            "ts": flow.index, "edge_id": eid, "flow": flow[eid].to_numpy(),
            "hour": start.hour + start.minute / 60, "weekend": (start.weekday >= 5).astype(int),
            "length_m": e.at[eid, "length_m"],
            "jload_mean": load[others].mean(axis=1).to_numpy(), "jload_max": load[others].max(axis=1).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


FEATURES = ["flow", "hour", "weekend", "length_m", "jload_mean", "jload_max"]


def estimate_load(flow: pd.DataFrame, load: pd.DataFrame, edges: pd.DataFrame, hidden: list[str],
                  flow_est: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """load: slots x edges, 15-minute mean load (hidden edges empty). Returns {method: slots x hidden
    edges} of 15-minute load_ratio."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    smooth_f = smooth(flow)
    visible_roads = [e for e in edges.edge_id[edges.kind == "road"] if e not in hidden]
    train = load_features(visible_roads, smooth_f, load, edges, hidden)
    train["y"] = load[visible_roads].to_numpy().T.ravel()
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
    model.fit(train[FEATURES], train.y)
    test = load_features(hidden, smooth_f.assign(**flow_est["M1"]), load, edges, hidden)
    m1 = pd.DataFrame(np.clip(model.predict(test[FEATURES]), 0, None).reshape(len(hidden), -1).T,
                      index=flow.index, columns=hidden)
    b0 = pd.DataFrame({h: load[visible_roads].mean(axis=1) for h in hidden})
    b1 = pd.DataFrame({h: load[opposite(h)] if opposite(h) not in hidden else np.nan for h in hidden},
                      index=load.index)
    return {"B0": b0, "B1": b1, "M1": m1}


def smooth(d: pd.DataFrame) -> pd.DataFrame:
    return d.rolling(SMOOTH_SLOTS, center=True, min_periods=1).mean()


MIN_CONGESTED = 50           # fewer congested 15-minute slots than this: recall is not reported


def evaluate(flow: pd.DataFrame, load: pd.DataFrame, edges: pd.DataFrame, hidden: list[str],
             scenario: str) -> tuple[pd.DataFrame, dict, dict]:
    """Estimate the hidden edges with every method and score them against what their cameras saw.
    Everything is compared as 15-minute means: a 5-minute count or a snapshot of the queue at the
    slot end is mostly noise (which car crosses the boundary, which signal phase is showing).
    Returns (estimate_eval rows, flow estimates, load estimates)."""
    _, a = junction_rows(edges)
    fixed = dict(zip(hidden, determined(a[hidden].to_numpy())))
    visible = [c for c in flow.columns if c not in hidden]
    load15 = smooth(load)
    f_est = estimate_flow(flow, edges, hidden)
    l_est = estimate_load(flow[visible + hidden], load15[visible].reindex(columns=load.columns), edges, hidden, f_est)
    true_f, true_l = smooth(flow[hidden]), load15[hidden]
    true_lv = level_of(true_l)
    hour = np.arange(len(flow)) // 12
    rows = []
    for method in f_est:
        lv = level_of(l_est[method])
        for h in hidden:
            fe, le = f_est[method][h], l_est[method][h]
            if fe.isna().all():
                continue                                   # B1 when the other direction is hidden too
            err = (fe - true_f[h]).abs()
            err_h = (fe.groupby(hour).sum() - flow[h].groupby(hour).sum()).abs()
            cong = true_lv[h].isin(CONGESTED)
            rows.append({"scenario": scenario, "method": method, "edge_id": h, "determined": int(fixed[h]),
                         "flow_mae": err.mean(), "flow_wape": err.sum() / true_f[h].sum(),
                         "flow_wape_1h": err_h.sum() / flow[h].sum(),
                         "load_mae": (le - true_l[h]).abs().mean(), "level_acc": (lv[h] == true_lv[h]).mean(),
                         "congested_slots": int(cong.sum()),
                         "congested_recall": lv[h][cong].isin(CONGESTED).mean() if cong.sum() >= MIN_CONGESTED
                         else np.nan})
    return pd.DataFrame(rows), f_est, l_est
