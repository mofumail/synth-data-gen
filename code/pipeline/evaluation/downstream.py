"""
DownstreamEvaluator + RecSysAdapter

TSTR (Train-Synthetic-Test-Real) pipeline.

adapter_cls is a CLASS REFERENCE, not a trained instance.
evaluate() instantiates and fits a fresh adapter on each call.
No trained state is held between calls - seed isolation guaranteed.

Orchestrator calls evaluate() three times per seed:
    evaluate(synth_T,    real_test, seed) -> TSTR-T   (transformer)
    evaluate(synth_M,    real_test, seed) -> TSTR-M   (markov baseline)
    evaluate(real_train, real_test, seed) -> TRTR     (interpretive upper bound)

RecSysAdapter interface:
    fit(interactions)              - train on a list of sessions
    predict(session_context, k)   - return top-k SKUs given prior events

PopularityAdapter is a global-popularity baseline (ignores session context).
GRU4RecAdapter is a session-based GRU recommender.
"""

from __future__ import annotations

import copy
import gc
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from config import (
    GRU4REC_VOCAB_K,
    GRU4REC_EMBED_DIM, GRU4REC_HIDDEN_DIM,
    GRU4REC_MAX_EPOCHS, GRU4REC_BATCH_SIZE, GRU4REC_LR,
    GRU4REC_PATIENCE, GRU4REC_LR_PATIENCE, GRU4REC_VAL_SPLIT, GRU4REC_LR_FACTOR,
)
from evaluation.data_classes import UtilityResult


# Abstract interface

class RecSysAdapter(ABC):
    """Model-agnostic interface. Any conforming recommender plugs in."""

    @abstractmethod
    def fit(self, interactions: List[List[dict]]) -> None:
        """Train on a list of interaction sessions (each = List[event dict])."""

    @abstractmethod
    def predict(self, session_context: List[dict], k: int) -> List[int]:
        """Return top-k recommended SKUs given the session context so far."""


# Popularity baseline (sanity check / fast lower bound)

class PopularityAdapter(RecSysAdapter):
    """
    Global-popularity baseline. Ignores session context entirely.
    Returns the globally most-popular SKUs from training.
    Useful as a sanity check: GRU4Rec should outperform this.
    """

    def __init__(self):
        self._top_items: List[int] = []

    def fit(self, interactions: List[List[dict]]) -> None:
        counts: Counter = Counter()
        for session in interactions:
            for ev in session:
                sku = ev.get("sku")
                if sku is not None:
                    counts[sku] += 1
        self._top_items = [sku for sku, _ in counts.most_common()]

    def predict(self, session_context: List[dict], k: int) -> List[int]:
        return self._top_items[:k]


# GRU4Rec

class _GRU4RecModel(nn.Module):
    """
    Single-layer GRU session-based recommender.
    Input : [B, T] local item indices (0 = padding)
    Output: [B, T, n_items+1] next-item logits
    """

    def __init__(self, n_items: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.gru      = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output   = nn.Linear(hidden_dim, n_items + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.item_emb(x))   # [B, T, hidden]
        return self.output(h)               # [B, T, n_items+1]


class GRU4RecAdapter(RecSysAdapter):
    """
    Session-based GRU4Rec recommender for TSTR evaluation.

    fit()     : builds a local item vocab from training interactions,
                then trains a small GRU on next-item prediction.
    predict() : encodes the session context through the GRU,
                returns top-k SKUs predicted at the last step.
    """

    def __init__(
        self,
        embed_dim:   int   = GRU4REC_EMBED_DIM,
        hidden_dim:  int   = GRU4REC_HIDDEN_DIM,
        max_epochs:  int   = GRU4REC_MAX_EPOCHS,
        batch_size:  int   = GRU4REC_BATCH_SIZE,
        lr:          float = GRU4REC_LR,
        patience:    int   = GRU4REC_PATIENCE,
        lr_patience: int   = GRU4REC_LR_PATIENCE,
        val_split:   float = GRU4REC_VAL_SPLIT,
        lr_factor:   float = GRU4REC_LR_FACTOR,
    ):
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.max_epochs  = max_epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.patience    = patience
        self.lr_patience = lr_patience
        self.val_split   = val_split
        self.lr_factor   = lr_factor
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model:   Optional[_GRU4RecModel] = None
        self._item2idx: dict = {}   # sku -> local index (1-based)
        self._idx2sku:  dict = {}   # local index -> sku

    def fit(self, interactions: List[List[dict]], label: str = "GRU4Rec") -> None:
        # --- Build local vocab from training items (capped to top-K by frequency) ---
        all_skus = [
            ev["sku"]
            for session in interactions
            for ev in session
            if ev.get("sku") is not None
        ]
        counts = Counter(all_skus)
        top_skus = [sku for sku, _ in counts.most_common(GRU4REC_VOCAB_K)]
        unique_skus = sorted(top_skus)
        self._item2idx = {sku: idx + 1 for idx, sku in enumerate(unique_skus)}
        self._idx2sku  = {idx: sku for sku, idx in self._item2idx.items()}
        n_items = len(unique_skus)
        print(f"\n  [{label}] GRU4Rec vocab: {n_items:,} items  (cap={GRU4REC_VOCAB_K:,}, raw unique={len(counts):,})", flush=True)

        # --- Convert sessions to local-index sequences ---
        sequences = self._sessions_to_sequences(interactions)
        if not sequences:
            return

        # --- Pad sequences and build dataset ---
        max_len  = max(len(s) for s in sequences)
        padded   = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        # Input = seq[:-1], target = seq[1:]
        inputs  = padded[:, :-1]   # [N, max_len-1]
        targets = padded[:, 1:]    # [N, max_len-1]

        dataset = TensorDataset(inputs, targets)

        # --- Val split ---
        val_size   = max(1, int(len(dataset) * self.val_split))
        train_size = len(dataset) - val_size
        if train_size < 1:
            train_ds, val_ds = dataset, None
        else:
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(0),
            )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=2, persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False,
                                  pin_memory=True, num_workers=2, persistent_workers=True) if val_ds else None

        # --- Model + optimiser + scheduler ---
        self._model = torch.compile(_GRU4RecModel(n_items, self.embed_dim, self.hidden_dim).to(self.device))
        optimizer   = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        scheduler   = ReduceLROnPlateau(optimizer, factor=self.lr_factor, patience=self.lr_patience)

        best_val_loss    = float("inf")
        best_state       = None
        patience_counter = 0

        import sys
        epoch_bar = tqdm(range(1, self.max_epochs + 1), desc=f"  [{label}]", unit="ep", leave=True, file=sys.stdout)
        for epoch in epoch_bar:
            # --- Train ---
            self._model.train()
            train_loss = 0.0
            for x_batch, y_batch in tqdm(train_loader, desc=f"    ep{epoch:02d} train", leave=False, file=sys.stdout):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                logits = self._model(x_batch)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), y_batch.view(B * T), ignore_index=0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- Validate + early stopping ---
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_batch, y_batch in tqdm(val_loader, desc=f"    ep{epoch:02d} val  ", leave=False, file=sys.stdout):
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        logits = self._model(x_batch)
                        B, T, V = logits.shape
                        val_loss += F.cross_entropy(logits.view(B * T, V), y_batch.view(B * T), ignore_index=0).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", patience=patience_counter)

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    best_state       = copy.deepcopy(self._model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        tqdm.write(f"    Early stopping at epoch {epoch} (patience={self.patience})")
                        break
            else:
                epoch_bar.set_postfix(train=f"{train_loss:.4f}")

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    def predict(self, session_context: List[dict], k: int) -> List[int]:
        if self._model is None or not session_context:
            return []

        # Convert context to local index sequence
        seq = [
            self._item2idx[ev["sku"]]
            for ev in session_context
            if ev.get("sku") is not None and ev["sku"] in self._item2idx
        ]
        if not seq:
            return []

        x = torch.tensor([seq], dtype=torch.long, device=self.device)  # [1, T]
        with torch.no_grad():
            logits = self._model(x)    # [1, T, V]
        last_logits = logits[0, -1]    # [V] - prediction at last step

        # Exclude padding index
        last_logits[0] = float("-inf")

        topk = torch.topk(last_logits, min(k, last_logits.size(0) - 1)).indices.tolist()
        return [self._idx2sku[idx] for idx in topk if idx in self._idx2sku]

    def cleanup(self) -> None:
        """Delete model and free GPU memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _sessions_to_sequences(self, interactions: List[List[dict]]) -> List[List[int]]:
        """Convert sessions to local-index item sequences (item-bearing events only)."""
        sequences = []
        for session in interactions:
            seq = [
                self._item2idx[ev["sku"]]
                for ev in session
                if ev.get("sku") is not None and ev["sku"] in self._item2idx
            ]
            if len(seq) >= 2:   # need at least input + one target
                sequences.append(seq)
        return sequences


# Downstream evaluator

class DownstreamEvaluator:
    """
    Instantiates a fresh RecSysAdapter per evaluate() call.
    Uses leave-one-out protocol: for each test session, the last
    item-bearing event is the ground truth; all prior item events = context.
    """

    def __init__(
        self,
        adapter_cls: Type[RecSysAdapter] = GRU4RecAdapter,
        k: int = 10,
        save_dir: Optional[str] = None,
        model_tag: str = "",
    ):
        self.adapter_cls = adapter_cls
        self.k           = k
        self.save_dir    = Path(save_dir) if save_dir else None
        self.model_tag   = model_tag

    def evaluate(
        self,
        train_sessions: List[List[dict]],
        real_test:      List[List[dict]],
        seed:           int,
        condition:      str = "TSTR",
    ) -> UtilityResult:
        """
        Instantiate a fresh adapter, fit on train_sessions, evaluate on real_test.

        Evaluation protocol (leave-one-out):
            For each test session with ≥2 item-bearing events:
                ground_truth = last item-bearing event's SKU
                context      = all item-bearing events before the last
                predictions  = adapter.predict(context, k=self.k)
                HR@K  += 1 if ground_truth in predictions
                NDCG@K += relevance-weighted rank score
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        adapter = self.adapter_cls()
        adapter.fit(train_sessions)

        # Optionally persist model for later inspection (overwrites per condition)
        if self.save_dir is not None and hasattr(adapter, "_model") and adapter._model is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            slug = condition.lower().replace("-", "_")
            fname = f"{self.model_tag}-{slug}-{len(train_sessions)}.pt"
            torch.save({"state_dict": adapter._model.state_dict(), "item2idx": adapter._item2idx}, self.save_dir / fname)
            print(f"    Saved GRU4Rec -> {self.save_dir / fname}", flush=True)

        test_pairs = self._build_test_pairs(real_test)
        if not test_pairs:
            return UtilityResult(condition=condition, hr_at_k=0.0, ndcg_at_k=0.0)

        # Filter OOV: ground truth items not in training vocab can never be predicted;
        # including them would deflate HR@K and NDCG@K for all conditions unfairly.
        n_before   = len(test_pairs)
        test_pairs = [(ctx, gt) for ctx, gt in test_pairs if gt in adapter._item2idx]
        n_oov      = n_before - len(test_pairs)
        oov_rate   = n_oov / n_before if n_before > 0 else 0.0
        print(f"    [{condition}] OOV filtered: {n_oov}/{n_before} ({oov_rate:.1%}) test pairs removed", flush=True)
        if not test_pairs:
            return UtilityResult(condition=condition, hr_at_k=0.0, ndcg_at_k=0.0, oov_rate=oov_rate)

        hr_list, ndcg_list = [], []
        for context, ground_truth_sku in test_pairs:
            predictions = adapter.predict(context, self.k)
            hr_list.append(self._hr_at_k(predictions, ground_truth_sku))
            ndcg_list.append(self._ndcg_at_k(predictions, ground_truth_sku))

        result = UtilityResult(
            condition = condition,
            hr_at_k   = float(np.mean(hr_list)),
            ndcg_at_k = float(np.mean(ndcg_list)),
            oov_rate  = oov_rate,
        )

        if hasattr(adapter, "cleanup"):
            adapter.cleanup()
        del adapter

        return result

    def evaluate_all_shared_vocab(
        self,
        conditions:     List[Tuple[str, List[List[dict]]]],
        real_test:      List[List[dict]],
        seed:           int,
        reference_cond: str = "TRTR",
    ) -> List[UtilityResult]:
        """
        Train all conditions, filter test pairs using the reference condition's vocab,
        then evaluate all conditions on identical test pairs for fair comparison.

        This prevents TSTR conditions from appearing artificially strong because their
        smaller vocab causes only easy (popular) test pairs to survive OOV filtering.

        Args:
            conditions:     List of (condition_name, train_sessions) tuples.
                            e.g. [("TSTR-T", synth_T), ("TSTR-M", synth_M), ("TRTR", real)]
            real_test:      Held-out test sessions (leave-one-out protocol).
            seed:           Random seed for adapter training.
            reference_cond: Which condition's vocab to use for shared OOV filtering.
                            Default "TRTR" (real-data vocab = fairest upper bound reference).
        Returns:
            List of UtilityResult in the same order as `conditions`.
        """
        # --- Train all adapters ---
        # Re-seed before EACH fit so the adapter's init/shuffle/etc. don't
        # depend on how much randomness earlier adapters consumed (vocab size,
        # epoch count via early stopping, etc.). Without this, TRTR drifts
        # whenever upstream synth data changes the TSTR-T/TSTR-M trajectories.
        adapters: dict = {}
        for condition, train_sessions in conditions:
            np.random.seed(seed)
            torch.manual_seed(seed)
            adapter = self.adapter_cls()
            adapter.fit(train_sessions, label=condition)
            adapters[condition] = adapter

            if self.save_dir is not None and hasattr(adapter, "_model") and adapter._model is not None:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                slug = condition.lower().replace("-", "_")
                fname = f"{self.model_tag}-{slug}-{len(train_sessions)}.pt"
                torch.save(
                    {"state_dict": adapter._model.state_dict(), "item2idx": adapter._item2idx},
                    self.save_dir / fname,
                )
                print(f"    Saved GRU4Rec -> {self.save_dir / fname}", flush=True)

        # --- Build shared vocabulary from the reference condition ---
        ref_adapter = adapters.get(reference_cond)
        if ref_adapter is None:
            # Fall back to intersection of all vocabs
            shared_vocab = set.intersection(*[set(a._item2idx.keys()) for a in adapters.values()])
            vocab_source  = "intersection"
        else:
            shared_vocab = set(ref_adapter._item2idx.keys())
            vocab_source  = reference_cond

        # --- Filter test pairs once using shared vocab ---
        all_pairs  = self._build_test_pairs(real_test)
        n_before   = len(all_pairs)
        shared_pairs = [(ctx, gt) for ctx, gt in all_pairs if gt in shared_vocab]
        n_oov      = n_before - len(shared_pairs)
        oov_rate   = n_oov / n_before if n_before > 0 else 0.0
        print(
            f"  [shared-vocab OOV filter ({vocab_source})] "
            f"{n_oov}/{n_before} ({oov_rate:.1%}) test pairs removed, "
            f"{len(shared_pairs):,} remain",
            flush=True,
        )

        # --- Evaluate all conditions on identical test pairs ---
        results = []
        for condition, _ in conditions:
            adapter = adapters[condition]
            if not shared_pairs:
                results.append(UtilityResult(condition=condition, hr_at_k=0.0, ndcg_at_k=0.0, oov_rate=oov_rate))
                continue

            hr_list, ndcg_list = [], []
            for context, ground_truth_sku in shared_pairs:
                predictions = adapter.predict(context, self.k)
                hr_list.append(self._hr_at_k(predictions, ground_truth_sku))
                ndcg_list.append(self._ndcg_at_k(predictions, ground_truth_sku))

            results.append(UtilityResult(
                condition = condition,
                hr_at_k   = float(np.mean(hr_list)),
                ndcg_at_k = float(np.mean(ndcg_list)),
                oov_rate  = oov_rate,
            ))
            print(
                f"    [{condition}] HR@{self.k}={results[-1].hr_at_k:.4f}  "
                f"NDCG@{self.k}={results[-1].ndcg_at_k:.4f}",
                flush=True,
            )

            if hasattr(adapter, "cleanup"):
                adapter.cleanup()

        return results

    def _hr_at_k(self, predictions: List[int], ground_truth: int) -> float:
        """HR@K: 1 if ground truth appears in top-K predictions."""
        return 1.0 if ground_truth in predictions[: self.k] else 0.0

    def _ndcg_at_k(self, predictions: List[int], ground_truth: int) -> float:
        """NDCG@K with binary relevance. Ideal DCG = 1 (single relevant item at rank 1)."""
        for rank, item in enumerate(predictions[: self.k], start=1):
            if item == ground_truth:
                return 1.0 / np.log2(rank + 1)
        return 0.0

    def _build_test_pairs(
        self,
        sessions: List[List[dict]],
    ) -> List[Tuple[List[dict], int]]:
        """
        Leave-one-out: ground truth = last item-bearing event's SKU.
        Context = all item-bearing events before the last.
        Sessions with <2 item-bearing events are skipped.
        Returns list of (context_events, ground_truth_sku).
        """
        pairs = []
        for session in sessions:
            item_evs = [e for e in session if e.get("sku") is not None]
            if len(item_evs) < 2:
                continue
            ground_truth = int(item_evs[-1]["sku"])
            context      = item_evs[:-1]
            pairs.append((context, ground_truth))
        return pairs
