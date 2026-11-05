"""Per-camera short-term forecaster (gradient boosting), this project's own: v1's forecaster went with its
clean-up (docs/setup.md). The same design as v1's:

  direct strategy   one model per (target, horizon): features at the origin t, the target at t + h
  features          <target>_now and lags (5 minutes each), rolling means, the other targets now, time of day
                    and day of week of the target time; subclasses add their own (forecast.NetworkForecaster)
  gaps              rows whose origin or target slot was not observed (`observed` = 0) are not learned from
  baselines         persistence (the value now) and same time yesterday (seasonal naive, 288 slots)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24]          # slots back from the origin (1 = the slot before now)
ROLLING = [3, 6, 12]                              # 15, 30, 60 minutes


@dataclass
class ForecasterSettings:
    targets: list[str]
    horizons: list[int]                           # slots ahead (3, 6, 12 = 15, 30, 60 minutes)
    quantiles: list[float] = field(default_factory=list)
    lags: list[int] = field(default_factory=lambda: list(LAGS))
    rolling: list[int] = field(default_factory=lambda: list(ROLLING))
    max_iter: int = 300
    learning_rate: float = 0.06
    min_rows: int = 200


class TrafficForecaster:
    backend = "gbm"

    def __init__(self, settings: ForecasterSettings, camera_id: str = ""):
        self.settings, self.camera_id = settings, camera_id
        self.models: dict[tuple[str, int], object] = {}

    def _features(self, series: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
        s = self.settings
        x = series[target]
        f = {f"{target}_now": x}
        for lag in s.lags:
            f[f"{target}_lag{lag}"] = x.shift(lag)
        for w in s.rolling:
            f[f"{target}_mean{w}"] = x.rolling(w, min_periods=1).mean()
        for other in s.targets:
            if other != target and other in series:
                f[f"{other}_now"] = series[other]
        when = series.index + pd.Timedelta(minutes=5 * h) - pd.Timedelta(minutes=5)   # start of the target slot
        tod = (when.hour * 60 + when.minute) / 1440
        f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
        f["weekday"] = when.weekday
        f["weekend"] = (when.weekday >= 5).astype(int)
        return pd.DataFrame(f, index=series.index)

    def _usable(self, series: pd.DataFrame, h: int) -> pd.Series:
        obs = series["observed"] if "observed" in series else pd.Series(1, index=series.index)
        return (obs == 1) & (obs.shift(-h) == 1)

    def fit(self, series: pd.DataFrame) -> "TrafficForecaster":
        from sklearn.ensemble import HistGradientBoostingRegressor

        s = self.settings
        for target in s.targets:
            for h in s.horizons:
                X = self._features(series, target, h)
                y = series[target].shift(-h)
                ok = self._usable(series, h) & y.notna() & series[target].notna()
                if ok.sum() < s.min_rows:
                    continue
                model = HistGradientBoostingRegressor(max_iter=s.max_iter, learning_rate=s.learning_rate,
                                                      random_state=0)
                self.models[(target, h)] = model.fit(X[ok], y[ok])
        return self

    def predict_frame(self, series: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
        """Forecast for t + h made at every origin t (indexed by the origin); NaN where there is no model."""
        model = self.models.get((target, h))
        if model is None:
            return pd.DataFrame({"point": np.nan}, index=series.index)
        point = np.clip(model.predict(self._features(series, target, h)), 0, None)
        return pd.DataFrame({"point": point}, index=series.index)


def persistence(series: pd.DataFrame, target: str, h: int) -> pd.Series:
    """The value at the origin, as the forecast for t + h."""
    return series[target]


def seasonal_naive(series: pd.DataFrame, target: str, h: int, season: int = 288) -> pd.Series:
    """The value one season (a day of 5-minute slots) before the target time."""
    return series[target].shift(season - h)
