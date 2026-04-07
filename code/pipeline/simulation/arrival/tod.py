"""
TODArrivalModel

Time-of-Day arrival model with weekday/weekend split.

Schema changes from original:
    visitorid  -> client_id
    timestamp  (ms int) -> timestamp (datetime, tz-naive)
    gap > 1800000 -> diff > pd.Timedelta(minutes=30)
"""

import os
import numpy as np
import pandas as pd
import joblib

from simulation.arrival.base import ArrivalModel
from config import CLEAN_PARQUET, SESSION_TIMEOUT_MIN


class TODArrivalModel(ArrivalModel):
    def __init__(self):
        self.rates      = None   # {"weekday": array[24], "weekend": array[24]}
        self.is_trained = False

    def fit(self) -> None:
        print(f"\nTraining TOD arrival model from {CLEAN_PARQUET}...")
        df = pd.read_parquet(CLEAN_PARQUET, columns=["client_id", "timestamp", "event_type"])
        df = df.sort_values(["client_id", "timestamp"])

        timeout  = pd.Timedelta(minutes=SESSION_TIMEOUT_MIN)
        diff     = df.groupby("client_id")["timestamp"].diff()
        arrivals = df[diff.isna() | (diff > timeout)].copy()

        arrivals["hour"]       = arrivals["timestamp"].dt.hour
        arrivals["is_weekend"] = arrivals["timestamp"].dt.dayofweek >= 5

        all_days       = pd.date_range(arrivals["timestamp"].min(), arrivals["timestamp"].max(), freq="D")
        n_weekend_days = sum(1 for d in all_days if d.dayofweek >= 5)
        n_weekday_days = len(all_days) - n_weekend_days

        counts = arrivals.groupby(["is_weekend", "hour"]).size().unstack(fill_value=0)

        self.rates = {
            "weekday": (counts.loc[False] / (n_weekday_days * 3600)).reindex(range(24), fill_value=0.0).values,
            "weekend": (counts.loc[True]  / (n_weekend_days * 3600)).reindex(range(24), fill_value=0.0).values,
        }

        self.is_trained = True
        print(f"  Trained on {n_weekday_days} weekdays and {n_weekend_days} weekends.")

    def get_next_arrival_delay(self, current_datetime: pd.Timestamp) -> float:
        if not self.is_trained:
            raise RuntimeError("TODArrivalModel not trained.")

        day_type = "weekend" if current_datetime.dayofweek >= 5 else "weekday"
        rate     = self.rates[day_type][current_datetime.hour]

        if rate <= 1e-12:
            return 3600.0
        return float(-np.log(np.random.random()) / rate)

    def save(self, path) -> None:
        joblib.dump({"rates": self.rates, "is_trained": self.is_trained}, path)
        print(f"  TOD model saved to {path}")

    def load(self, path) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        data = joblib.load(path)
        self.rates      = data["rates"]
        self.is_trained = data["is_trained"]
        print(f"  TOD model loaded from {path}")


if __name__ == "__main__":
    from config import MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = TODArrivalModel()
    model.fit()
    model.save(MODEL_DIR / "tod_model.pkl")
