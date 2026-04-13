"""
SimulationOrchestrator

Drives the full simulation loop.

    - Schema: client_id/sku 
    - Paths via config.py 
    - user_history_buffer added 
    - generate_session call passes history argument
    - Imports from module structure 

"""

import uuid
import time
import multiprocessing as mp
from collections import deque
from typing import Dict

import pandas as pd

from config import MODEL_DIR, EVAL_MODEL_SUBDIR, OUTPUT_DIR, SYNTH_DIR
from simulation.arrival.mmpp import MMPPArrivalModel
from simulation.arrival.tod  import TODArrivalModel
from simulation.identity.sampler import SimpleIdentitySampler
from simulation.generator.session_generator import SessionGenerator
from simulation.validity import ValidityLayer
from evaluation.reference import ReferenceStore

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_WINDOW  = 50                                   # H: past events kept per user
REF_STORE_PATH  = OUTPUT_DIR / "reference_store.joblib"


class SimulationOrchestrator:
    def __init__(self, arrival_model_type: str = "tod"):
        print(f"\nInitializing SimulationOrchestrator (arrival: {arrival_model_type})")
        self.model_type = arrival_model_type.lower()

        if self.model_type == "mmpp":
            self.arrival_model = MMPPArrivalModel()
            self.arrival_model.load(MODEL_DIR / "mmpp_model.pkl")
        elif self.model_type == "tod":
            self.arrival_model = TODArrivalModel()
            self.arrival_model.load(MODEL_DIR / "tod_model.pkl")
        else:
            raise ValueError(f"Unknown arrival model type: {arrival_model_type}")

        self.identity_factory = SimpleIdentitySampler()
        self.identity_factory.load(MODEL_DIR / "identity_sampler.pkl")

        if REF_STORE_PATH.exists():
            ref_store = ReferenceStore.load(REF_STORE_PATH)
            legal_bigrams = ref_store.legal_bigrams_set()
            print(f"  ValidityLayer: {len(legal_bigrams)} legal bigrams loaded from reference store")
        else:
            legal_bigrams = set()
            print(f"  ValidityLayer: WARNING - reference store not found at {REF_STORE_PATH}. "
                  "Run evaluate.py first to build it. Bigram check disabled.")
        self.validity_layer = ValidityLayer(legal_bigrams=legal_bigrams)

        self.session_generator = SessionGenerator(
            model_path=str(EVAL_MODEL_SUBDIR / "model.pt"),
            validity_layer=self.validity_layer,
        )

        self.event_buffer: list = []
        self.user_history_buffer: Dict[int, deque] = {}

    def _get_history(self, client_id: int) -> list:
        if client_id not in self.user_history_buffer:
            self.user_history_buffer[client_id] = deque(maxlen=HISTORY_WINDOW)
        return list(self.user_history_buffer[client_id])

    def _update_history(self, client_id: int, events: list) -> None:
        if client_id not in self.user_history_buffer:
            self.user_history_buffer[client_id] = deque(maxlen=HISTORY_WINDOW)
        self.user_history_buffer[client_id].extend(events)

    def save_batch(self) -> None:
        if not self.event_buffer:
            return

        df        = pd.DataFrame(self.event_buffer)
        batch_id  = uuid.uuid4().hex[:8]
        file_path = SYNTH_DIR / f"synthetic_batch_{batch_id}.parquet"
        df.to_parquet(file_path, index=False)
        print(f"\n  Batch saved: {file_path} ({len(df):,} events)")
        self.event_buffer = []

    def run(self, start_date: str = "2022-10-01", days_to_simulate: int = 31) -> None:
        current_dt = pd.Timestamp(start_date)
        end_dt     = current_dt + pd.Timedelta(days=days_to_simulate)
        start_ts   = current_dt
        arrival_count = 0

        tasks_in_flight = mp.Value("i", 0)
        pool = mp.Pool(processes=mp.cpu_count())

        def task_callback(result):
            nonlocal arrival_count
            client_id, session_events = result
            self.event_buffer.extend(session_events)
            self._update_history(client_id, session_events)
            with tasks_in_flight.get_lock():
                tasks_in_flight.value -= 1
            if arrival_count % 1000 == 0:
                print(f"\n  {arrival_count:,} sessions submitted | buffer: {len(self.event_buffer):,}")

        def error_callback(e):
            print(f"\n  Task error: {e}")

        print(f"\nSimulation: {current_dt} -> {end_dt}")

        while current_dt < end_dt:
            if self.model_type == "tod":
                delay = self.arrival_model.get_next_arrival_delay(current_dt)
            else:
                elapsed = (current_dt - start_ts).total_seconds()
                delay   = self.arrival_model.get_next_arrival_delay(elapsed)

            current_dt += pd.Timedelta(seconds=delay)
            if current_dt >= end_dt:
                break

            identity  = self.identity_factory.generate_identity()
            client_id = identity["client_id"]
            sku       = identity["sku"]
            history   = self._get_history(client_id)

            pool.apply_async(
                _generate_session_worker,
                args=(self.session_generator, client_id, sku, current_dt, history),
                callback=task_callback,
                error_callback=error_callback,
            )
            arrival_count += 1
            with tasks_in_flight.get_lock():
                tasks_in_flight.value += 1

            while tasks_in_flight.value > mp.cpu_count() * 2:
                time.sleep(0.05)

            if len(self.event_buffer) >= 10_000:
                self.save_batch()

        pool.close()
        pool.join()
        self.save_batch()
        print(f"\nSimulation complete. Total arrivals: {arrival_count:,}")


def _generate_session_worker(generator, client_id, sku, start_dt, history):
    """Top-level function so it's picklable by multiprocessing."""
    events = generator.generate_session(client_id, sku, start_dt, history)
    return client_id, events


if __name__ == "__main__":
    orch = SimulationOrchestrator(arrival_model_type="tod")
    orch.run(start_date="2022-10-01", days_to_simulate=31)
