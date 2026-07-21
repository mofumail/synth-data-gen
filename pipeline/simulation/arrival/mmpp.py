"""
MMPPArrivalModel

Markov-Modulated Poisson Process for session arrival timing.

Schema changes from original:
    visitorid  -> client_id
    timestamp  (ms int) -> timestamp (datetime, tz-naive)
    gap > 1800000 -> diff > pd.Timedelta(minutes=30)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans

from simulation.arrival.base import ArrivalModel
from config import CLEAN_PARQUET, SESSION_TIMEOUT_MIN


class MMPPArrivalModel(ArrivalModel):
    def __init__(self, num_states: int = 6, bin_minutes: int = 30):
        self.num_states   = num_states
        self.bin_minutes  = bin_minutes
        self.lambdas      = None
        self.transition_matrix = None
        self.current_state     = 0
        self.is_trained        = False

        self.last_state_switch_time = 0
        self.min_state_duration     = 15 * 60   # seconds

    def fit(self) -> None:
        print(f"\nTraining {self.num_states}-state MMPP from {CLEAN_PARQUET}...")
        df = pd.read_parquet(CLEAN_PARQUET, columns=["client_id", "timestamp", "event_type"])
        df = df.sort_values(["client_id", "timestamp"])

        timeout = pd.Timedelta(minutes=SESSION_TIMEOUT_MIN)
        diff = df.groupby("client_id")["timestamp"].diff()
        arrivals = df[diff.isna() | (diff > timeout)].copy()
        print(f"  Identified {len(arrivals):,} session starts from {len(df):,} events.")

        bin_width = f"{self.bin_minutes}min"
        counts = arrivals.set_index("timestamp").resample(bin_width).size()

        data   = counts.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.num_states, n_init=10, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_

        raw_lambdas = kmeans.cluster_centers_.flatten() / (self.bin_minutes * 60.0)
        idx_sort    = np.argsort(raw_lambdas)
        self.lambdas = raw_lambdas[idx_sort]

        rank_map      = {old: new for new, old in enumerate(idx_sort)}
        sorted_labels = np.vectorize(rank_map.get)(labels)

        trans_mat = np.zeros((self.num_states, self.num_states))
        for i in range(len(sorted_labels) - 1):
            trans_mat[sorted_labels[i], sorted_labels[i + 1]] += 1

        row_sums = trans_mat.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            trans_mat, row_sums,
            out=np.zeros_like(trans_mat),
            where=row_sums != 0,
        )

        self.is_trained = True
        print(f"\n  States (arrivals/sec):")
        for i, lam in enumerate(self.lambdas):
            print(f"    State {i}: {lam:.6f} arr/sec  ({lam * 3600:.1f} arr/hr)")

    def get_next_arrival_delay(self, current_sim_time_seconds: float) -> float:
        if not self.is_trained:
            raise RuntimeError("MMPPArrivalModel not trained.")

        time_since_switch = current_sim_time_seconds - self.last_state_switch_time
        if time_since_switch > self.min_state_duration:
            self.current_state = np.random.choice(
                self.num_states,
                p=self.transition_matrix[self.current_state],
            )
            self.last_state_switch_time = current_sim_time_seconds

        rate = self.lambdas[self.current_state]
        if rate <= 1e-12:
            return 3600.0
        return float(-np.log(np.random.random()) / rate)

    def save(self, path) -> None:
        if not self.is_trained:
            print("Cannot save: model not trained.")
            return
        joblib.dump({
            "lambdas":           self.lambdas,
            "transition_matrix": self.transition_matrix,
            "num_states":        self.num_states,
            "bin_minutes":       self.bin_minutes,
            "is_trained":        self.is_trained,
        }, path)
        print(f"  MMPP model saved to {path}")

    def load(self, path) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        data = joblib.load(path)
        self.lambdas           = data["lambdas"]
        self.transition_matrix = data["transition_matrix"]
        self.num_states        = data["num_states"]
        self.bin_minutes       = data.get("bin_minutes", 30)
        self.is_trained        = data["is_trained"]
        self.current_state     = 0
        self.last_state_switch_time = 0
        print(f"  MMPP model loaded from {path}")


if __name__ == "__main__":
    from config import MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = MMPPArrivalModel(num_states=6, bin_minutes=30)
    model.fit()
    model.save(MODEL_DIR / "mmpp_model.pkl")
