"""
In-training TSTR-T probe.

Drives `is_best` checkpoint selection during training: every N epochs, freeze
the model, generate synthetic sessions for K seeds, fit GRU4Rec on each, and
score HR@K against the real val split. Mean HR across seeds is the signal.

This is the actual generation-time fidelity proxy that val CE can't see, see
notebooks/why-val-loss-isnt-fidelity.md for the rationale.

Designed for one call per probe-epoch from train.py. The train-time
SessionTransformer should be moved to CPU before calling so the GRU4Rec runs
have GPU headroom.
"""
from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from evaluation.downstream import DownstreamEvaluator
from evaluation.reference import ReferenceStore
from simulation.generator.session_generator import SessionGenerator
from simulation.validity import ValidityLayer


def run_tstr_probe(
    candidate_path: Path,
    val_split: List[List[dict]],
    n_sessions: int,
    seeds: List[int],
    ref_store: ReferenceStore,
    identity_sampler_path: str,
    k: int = 10,
    device: str = "cuda",
) -> Dict:
    """
    For each seed: generate n_sessions synthetic sessions from the candidate
    checkpoint, fit a fresh GRU4Rec on them, evaluate HR@k / NDCG@k on
    val_split. Returns aggregated stats.

    The SessionGenerator's underlying transformer is freed between seeds and
    reloaded as needed so we don't hold two transformer copies on GPU at once
    while GRU4Rec is also using it.
    """
    valid_transitions = defaultdict(set)
    for a, b in ref_store.legal_bigrams:
        valid_transitions[a].add(b)
    validity_layer = ValidityLayer(legal_bigrams=ref_store.legal_bigrams_set())

    down_ev = DownstreamEvaluator(k=k, save_dir=None, model_tag="tstr-probe")

    hrs: List[float]   = []
    ndcgs: List[float] = []
    for seed in seeds:
        gen = SessionGenerator(
            model_path            = str(candidate_path),
            validity_layer        = validity_layer,
            valid_transitions     = valid_transitions,
            identity_sampler_path = identity_sampler_path,
            device                = device,
        )
        sessions_all = gen.generate(
            n_sessions=n_sessions, seed=seed, apply_constraints=True,
        )
        sessions = [s for s in sessions_all if s]
        non_empty = len(sessions)
        print(
            f"  [probe seed={seed}] generated {non_empty:,}/{n_sessions:,} non-empty "
            f"({100*non_empty/n_sessions:.1f}%)",
            flush=True,
        )

        # Free generator GPU memory before GRU4Rec needs it
        if gen._model is not None:
            try:
                gen._model.cpu()
            except Exception:
                pass
            del gen._model
            gen._model = None
        del gen
        gc.collect()
        torch.cuda.empty_cache()

        result = down_ev.evaluate(
            train_sessions = sessions,
            real_test      = val_split,
            seed           = seed,
            condition      = f"TSTR-T-probe-s{seed}",
        )
        hrs.append(result.hr_at_k)
        ndcgs.append(result.ndcg_at_k)
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "hr_mean":      float(np.mean(hrs)),
        "hr_std":       float(np.std(hrs)),
        "ndcg_mean":    float(np.mean(ndcgs)),
        "ndcg_std":     float(np.std(ndcgs)),
        "hr_per_seed":  list(zip(seeds, [float(h) for h in hrs])),
        "ndcg_per_seed": list(zip(seeds, [float(n) for n in ndcgs])),
    }
