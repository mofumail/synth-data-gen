"""
SessionGenerator

Wraps SessionTransformer inference and the ValidityLayer post-filter.
Implements GeneratorInterface for use by the EvaluationOrchestrator.

Pickling note: SessionGenerator is called inside multiprocessing workers by
SimulationOrchestrator. Keep state minimal - no open file handles. The model
is loaded lazily on first use inside the worker process.

See: mermaid/new/PROPOSED_Level3.md
     mermaid/new/PROPOSED_SLC1.md
     mermaid/Level4_EvaluationModule.md (GeneratorInterface)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import DS_START, HISTORY_WINDOW, VOCAB_K
from simulation.validity import ValidityLayer
from simulation.generator.session_transformer import SessionTransformer


class SessionGenerator:
    """
    Wraps SessionTransformer inference + ValidityLayer.

    Args:
        model_path: path to a SessionTransformer checkpoint (.pt)
        validity_layer: ValidityLayer built from training bigrams
        valid_transitions: dict passed to SessionTransformer.load() for mask
        identity_sampler_path: optional path to a SimpleIdentitySampler pickle;
            used by generate() when no explicit identity is given
        device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model_path: str,
        validity_layer: ValidityLayer,
        valid_transitions: dict = None,
        identity_sampler_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 256,
    ):
        self.model_path            = str(model_path)
        self.validity_layer        = validity_layer
        self.valid_transitions     = valid_transitions or {}
        self.identity_sampler_path = identity_sampler_path
        self.device                = device
        self.batch_size            = batch_size
        self._model                = None    # loaded lazily
        self._sampler              = None    # loaded lazily
        # Accumulates generated sessions per user across generate() calls.
        # Cleared between evaluation seeds via reset_registry().
        self._user_registry: Dict[int, List[dict]] = {}

    # Lazy loading

    def _load_model(self) -> None:
        """Load SessionTransformer from checkpoint (called once per process)."""
        self._model = SessionTransformer.load(
            self.model_path,
            valid_transitions=self.valid_transitions,
            device=self.device,
        )
        # Attach sku2idx / idx2sku for correct item encoding/decoding in infer()
        try:
            import joblib
            from ingestion.dataset import SKU2IDX_PATH
            sku2idx = joblib.load(SKU2IDX_PATH)
            self._model._sku2idx = sku2idx
            self._model._idx2sku = {v: k for k, v in sku2idx.items()}
        except Exception:
            pass   # fall back to sku+1 offset if mapping not found

    def _load_sampler(self) -> None:
        """Optionally load IdentityFactory for generate()."""
        if self.identity_sampler_path is None:
            return
        import joblib
        self._sampler = joblib.load(self.identity_sampler_path)

    # Single-session inference

    def generate_session(
        self,
        client_id: int,
        sku: int,
        start_dt,
        history: Optional[List[dict]] = None,
        max_steps: int = 50,
        temperature: float = 1.0,
        apply_constraints: bool = True,
    ) -> List[dict]:
        """
        Autoregressive inference for one session.

        If apply_constraints=True the session passes through ValidityLayer.
        Invalid sessions are returned as empty lists (caller handles).

        Args:
            client_id: user id seed
            sku: item id seed
            start_dt: session start datetime
            history: past event dicts for cross-session conditioning
            max_steps: max events before forced stop
            temperature: sampling temperature
            apply_constraints: run ValidityLayer post-filter

        Returns:
            List[dict] with keys client_id, event_type, sku, timestamp.
            Empty list if apply_constraints=True and session fails validity.
        """
        if self._model is None:
            self._load_model()

        events = self._model.infer(
            client_id=client_id,
            sku=sku,
            start_dt=start_dt,
            history=history,
            max_steps=max_steps,
            temperature=temperature,
        )

        if apply_constraints and events:
            if not self.validity_layer.validate(events):
                return []

        return events

    def reset_registry(self) -> None:
        """Clear accumulated user history. Call between evaluation seeds."""
        self._user_registry.clear()

    # GeneratorInterface - used by EvaluationOrchestrator

    def generate(
        self,
        n_sessions: int,
        seed: int,
        apply_constraints: bool = True,
    ) -> List[List[dict]]:
        """
        GeneratorInterface implementation.

        Generates n_sessions synthetic sessions with deterministic seeding.
        apply_constraints=False returns raw sessions (for pre-validity-filter
        violation rate measurement).

        Sessions are seeded with (client_id, sku) from the identity sampler if
        one was provided at construction; otherwise random integers are used.

        Returns:
            List of sessions; each session is List[dict].
            Empty sessions (validity failure) are included as [] - the caller
            (EvaluationOrchestrator) filters or counts them.
        """
        if self._model is None:
            self._load_model()
        if self._sampler is None and self.identity_sampler_path is not None:
            self._load_sampler()

        rng_py  = random.Random(seed)
        rng_np  = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Spread session start times uniformly over a 30-day window
        start_base = pd.Timestamp(DS_START)
        window_s   = 30 * 24 * 3600

        # Build all (client_id, sku, start_dt, history) inputs up front
        batch_inputs: List[tuple] = []
        for _ in range(n_sessions):
            if self._sampler is not None:
                identity = self._sampler.generate_identity()
                cid      = identity["client_id"]
                item     = identity["sku"]
            else:
                cid  = int(rng_np.integers(1, 10_000_000))
                item = int(rng_np.integers(0, VOCAB_K))

            offset_s = rng_py.randint(0, window_s)
            start_dt = start_base + pd.Timedelta(seconds=offset_s)
            past     = self._user_registry.get(cid)
            history  = past[-HISTORY_WINDOW:] if past else None
            batch_inputs.append((cid, item, start_dt, history))

        # Generate in batches - all sessions in a batch run in parallel on GPU
        sessions: List[List[dict]] = []
        n_batches = (n_sessions + self.batch_size - 1) // self.batch_size
        for start in tqdm(range(0, n_sessions, self.batch_size), total=n_batches, desc="  Transformer gen", unit="batch", leave=False, file=__import__("sys").stdout):
            chunk   = batch_inputs[start : start + self.batch_size]
            results = self._model.infer_batch(chunk)
            if apply_constraints:
                results = [
                    s if (s and self.validity_layer.validate(s)) else []
                    for s in results
                ]
            # Accumulate generated events into per-user registry
            for (cid, _, _, _), session in zip(chunk, results):
                if session:
                    self._user_registry.setdefault(cid, []).extend(session)
            sessions.extend(results)

        return sessions
