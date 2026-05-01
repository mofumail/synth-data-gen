"""
SessionTransformer training script.

Four factored cross-entropy losses:
    action_loss   : CE over all real positions
    category_loss : CE over item-bearing positions (next-item category)
    item_loss     : hierarchical CE, masked to the target category's SKU pool
    temporal_loss : CE over all real positions

All hyperparameters are read from config.yaml.

Usage: PYTHONPATH=. uv run python train.py

Each run writes to output/models/<MODEL_NAME>_<DDMMYY-HH-MM-SS>/; within a
run, per-epoch "is_best" checkpoints overwrite model.pt / model.yaml inside
that same folder. A config.yaml snapshot is written alongside the checkpoint.
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import yaml
import wandb
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    MODEL_NAME, MODEL_DIR, VOCAB_K, N_TEMPORAL_BINS, HISTORY_WINDOW,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH,
    TRAIN_LR, TRAIN_D_MODEL, TRAIN_N_LAYERS, TRAIN_N_HEADS,
    TRAIN_MAX_SESSIONS, TRAIN_NUM_WORKERS, TRAIN_PATIENCE,
    N_CATEGORIES, CAT2IDX_PATH, OUTPUT_DIR,
    SVDPQ_ENABLED, SVDPQ_T, SVDPQ_V, SVDPQ_LABEL_SMOOTHING,
    ITEM_LOSS_FREQ_WEIGHT_ALPHA, CAT_LOGIT_ADJ_TAU,
    IS_BEST_EVAL_ENABLED, IS_BEST_EVAL_EVERY_N_EPOCHS,
    IS_BEST_EVAL_N_SESSIONS, IS_BEST_EVAL_SEEDS, IS_BEST_EVAL_K,
    LOSS_WEIGHTS_ENABLED, LOSS_WEIGHTS,
)
from ingestion.dataset import (
    InteractionGenerator, get_cat_vocab, get_sku2idx, get_sku_properties,
    get_sku_tokens, get_vocab_stats,
)
import gc
import numpy as np
from simulation.generator.heads import ITEM_BEARING_IDX, SVDPQItemHead
from simulation.generator.session_transformer import SessionTransformer


def build_target_mask(lengths: torch.Tensor, T_out: int, device) -> torch.Tensor:
    """
    True at positions that are REAL targets (not padding).
    For session of length L, real output positions are 0..L-2
    (predicting tokens 1..L-1 from inputs 0..L-2).
    """
    positions = torch.arange(T_out, device=device).unsqueeze(0)   # [1, T_out]
    return positions < (lengths.to(device).unsqueeze(1) - 1)       # [B, T_out]


def train(comment: str | None = None):
    # Stamp the run once at start so every epoch writes into the same folder.
    run_stamp  = datetime.now().strftime("%d%m%y-%H-%M-%S")
    suffix     = f"_{comment}" if comment else ""
    run_name   = f"{MODEL_NAME}_{run_stamp}{suffix}"
    run_subdir = MODEL_DIR / run_name
    run_subdir.mkdir(parents=True, exist_ok=True)
    # Record the run folder so main.py can find it after training even if
    # config.yaml (and thus MODEL_NAME) changed mid-run.
    (MODEL_DIR / ".last_run").write_text(run_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Run:    {run_name}")
    print(f"Config: epochs={TRAIN_EPOCHS}, batch={TRAIN_BATCH_SIZE}, "
          f"d_model={TRAIN_D_MODEL}, layers={TRAIN_N_LAYERS}, heads={TRAIN_N_HEADS}, "
          f"lr={TRAIN_LR}, max_sessions={TRAIN_MAX_SESSIONS}, n_categories={N_CATEGORIES}, "
          f"patience={TRAIN_PATIENCE}")

    wandb.init(
        project="thesis-session-transformer",
        name=run_name,
        config={
            "model":           run_name,
            "epochs":          TRAIN_EPOCHS,
            "batch_size":      TRAIN_BATCH_SIZE,
            "max_length":      TRAIN_MAX_LENGTH,
            "lr":              TRAIN_LR,
            "d_model":         TRAIN_D_MODEL,
            "n_layers":        TRAIN_N_LAYERS,
            "n_heads":         TRAIN_N_HEADS,
            "vocab_k":         VOCAB_K,
            "n_temporal_bins": N_TEMPORAL_BINS,
            "history_window":  HISTORY_WINDOW,
            "max_sessions":    TRAIN_MAX_SESSIONS,
            "n_categories":    N_CATEGORIES,
            "patience":        TRAIN_PATIENCE,
            "item_loss_freq_weight_alpha": ITEM_LOSS_FREQ_WEIGHT_ALPHA,
            "cat_logit_adj_tau":           CAT_LOGIT_ADJ_TAU,
            "is_best_eval_enabled":        IS_BEST_EVAL_ENABLED,
            "is_best_eval_every_n_epochs": IS_BEST_EVAL_EVERY_N_EPOCHS,
            "is_best_eval_n_sessions":     IS_BEST_EVAL_N_SESSIONS,
            "is_best_eval_seeds":          list(IS_BEST_EVAL_SEEDS),
            "is_best_eval_k":              IS_BEST_EVAL_K,
            "loss_weights_enabled":        LOSS_WEIGHTS_ENABLED,
            "loss_weights":                list(LOSS_WEIGHTS),
        },
    )

    # All metrics should use epoch as x-axis
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*",   step_metric="epoch")

    # Dataset
    _t_train_load = time.perf_counter()
    print(f"\nLoading dataset (split=train, max_sessions={TRAIN_MAX_SESSIONS}) ...")
    gen = InteractionGenerator(
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=TRAIN_NUM_WORKERS,
        max_sessions=TRAIN_MAX_SESSIONS,
    )
    print(f"{len(gen.dataset):,} training sessions | {len(gen):,} batches/epoch")
    print(f"  >> train load: {time.perf_counter() - _t_train_load:.2f}s")

    _t_val_load = time.perf_counter()
    print(f"\nLoading validation dataset ...")
    val_gen = InteractionGenerator(
        split="val",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=TRAIN_NUM_WORKERS,
        max_sessions=None,        # always use full val set
    )
    print(f"{len(val_gen.dataset):,} val sessions | {len(val_gen):,} batches")
    print(f"  >> val load:   {time.perf_counter() - _t_val_load:.2f}s")
    print(f"  >> total dataset load: {time.perf_counter() - _t_train_load:.2f}s")

    # Category vocab: load per-category SKU pools as GPU tensors. Used to build
    # the masked softmax so the item-head loss normalizes only over SKUs in the
    # target category (the whole point of the factorization).
    if not CAT2IDX_PATH.exists():
        raise RuntimeError(
            "cat2idx.joblib not found. Run preprocess.py first to build category vocab."
        )
    sku2idx = get_sku2idx()
    _, cat_pools_np = get_cat_vocab(sku2idx)
    cat_sku_pools_tensors = [
        torch.as_tensor(p, dtype=torch.long, device=device) for p in cat_pools_np
    ]
    nonempty = sum(1 for p in cat_sku_pools_tensors if p.numel() > 0)
    avg_pool = sum(p.numel() for p in cat_sku_pools_tensors) / max(nonempty, 1)
    print(f"  {N_CATEGORIES} categories ({nonempty} non-empty pools, avg pool size {avg_pool:.0f})")

    # SKU property tables (price + quantized name) — input-side features looked
    # up by SKU index inside the model.
    sku_props = get_sku_properties(sku2idx)
    print(f"  SKU aux features: price + name")

    # SVD-PQ tokens (optional output-side item representation)
    sku_tokens = None
    if SVDPQ_ENABLED:
        sku_tokens = get_sku_tokens()
        print(f"  SVD-PQ tokens: shape={sku_tokens.shape} dtype={sku_tokens.dtype} "
              f"(t={SVDPQ_T}, v={SVDPQ_V})")

    # Popularity-bias mitigation priors (built once per run from cached stats).
    #
    # A. Per-event inverse-frequency weights for the item CE loss.
    #    w_sku = (count + 1) ** -alpha, normalized so E_data[w] = 1 (preserves
    #    multi-task loss balance). alpha=0 -> all-ones -> identical to vanilla CE.
    item_freq_weights = None
    if ITEM_LOSS_FREQ_WEIGHT_ALPHA > 0.0:
        _, sku_counts = get_vocab_stats()                     # [VOCAB_K], counts[0]=PAD=0
        c = sku_counts.astype(np.float64)
        w = (c + 1.0) ** (-ITEM_LOSS_FREQ_WEIGHT_ALPHA)
        # Renormalize so the expected weight under the training event distribution is 1.
        # E_data[w] = sum(c * w) / sum(c).
        denom = float((c * w).sum() / max(c.sum(), 1.0))
        if denom > 0:
            w = w / denom
        w[0] = 0.0                                            # PAD never targeted
        item_freq_weights = torch.as_tensor(w, dtype=torch.float32, device=device)
        # Sanity print
        head = sku_counts.argsort()[::-1][:5]
        print(
            f"  Item-loss inverse-frequency weights ON (alpha={ITEM_LOSS_FREQ_WEIGHT_ALPHA}): "
            f"head-5 mean w={float(w[head].mean()):.3f}, "
            f"tail-5 sample w={float(w[(c > 0).nonzero()[0][-5:]].mean()):.3f}"
        )

    # B. Category-head logit adjustment prior.
    #    train_logits = raw_logits + tau * log(prior). Built from cached
    #    categories.npy (item-bearing positions only — index 0 is PAD).
    cat_logit_adj = None
    if CAT_LOGIT_ADJ_TAU > 0.0:
        cat_flat = np.asarray(gen.dataset._categories_flat)   # mmap'd; bincount realizes it
        cat_counts = np.bincount(cat_flat, minlength=N_CATEGORIES).astype(np.float64)
        cat_counts[0] = 0.0                                   # PAD is never a target
        prior = cat_counts / max(cat_counts.sum(), 1.0)
        # Use additive smoothing only on zero-count categories so non-zero priors
        # are preserved exactly (cleaner Bayes-optimal interpretation). Empty cats
        # are masked at inference anyway, so their training logits don't matter.
        eps = 1.0 / max(cat_counts.sum(), 1.0)
        log_prior = np.log(np.where(prior > 0, prior, eps))
        log_prior[0] = 0.0                                    # never used
        cat_logit_adj = (
            CAT_LOGIT_ADJ_TAU
            * torch.as_tensor(log_prior, dtype=torch.float32, device=device)
        )
        nonzero = int((cat_counts > 0).sum())
        print(
            f"  Category logit-adjustment ON (tau={CAT_LOGIT_ADJ_TAU}): "
            f"{nonzero}/{N_CATEGORIES} categories have nonzero prior, "
            f"max log_prior={float(log_prior.max()):.2f}, "
            f"min log_prior(nonzero)={float(log_prior[cat_counts > 0].min()):.2f}"
        )

    # Model
    model = SessionTransformer(
        vocab_size=VOCAB_K,
        d_model=TRAIN_D_MODEL,
        n_layers=TRAIN_N_LAYERS,
        n_heads=TRAIN_N_HEADS,
        n_temporal_bins=N_TEMPORAL_BINS,
        window_H=HISTORY_WINDOW,
        n_categories=N_CATEGORIES,
        sku_price_table=sku_props["price"],
        sku_name_table =sku_props["name"],
        sku_tokens_table=sku_tokens,
        svdpq_t=SVDPQ_T if SVDPQ_ENABLED else 0,
        svdpq_v=SVDPQ_V if SVDPQ_ENABLED else 0,
        svdpq_label_smoothing=SVDPQ_LABEL_SMOOTHING if SVDPQ_ENABLED else 0.0,
    ).to(device)
    # Capture the uncompiled item_head BEFORE torch.compile so its .loss /
    # .hierarchical_loss reaches the real module rather than going through the
    # OptimizedModule wrapper.
    item_head = model.item_head
    if isinstance(item_head, SVDPQItemHead):
        item_head.register_sku_tokens(
            torch.as_tensor(sku_tokens, dtype=torch.long, device=device)
        )
    #test
    else:
        item_head.register_sku_cat_map(cat_sku_pools_tensors, VOCAB_K)
    # torch.compile disabled: hits an inductor tiling_utils assertion on this
    # model graph (PyTorch bug, not ours). bf16 alone is fast enough here.
    # model = torch.compile(model, dynamic=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS * len(gen), eta_min=1e-6)

    best_loss   = float("inf")        # diagnostic; only drives is_best when probe disabled
    best_hr     = float("-inf")       # primary signal when probe enabled
    epochs_no_improve  = 0            # counts {val_loss epochs OR probes} since last improvement
    probes_no_improve  = 0
    item_action_ids = torch.tensor(sorted(ITEM_BEARING_IDX), dtype=torch.long, device=device)

    # In-training TSTR-T probe wiring. Loaded once so the train loop never
    # re-pays the cost. val_real_split is filtered to the current vocab so OOV
    # ground-truth items don't auto-fail HR for every condition equally
    # (matches what evaluate.py does for end-of-run TSTR).
    val_real_split             = None
    probe_ref_store            = None
    probe_identity_sampler_path = None
    if IS_BEST_EVAL_ENABLED:
        from evaluation.reference import (
            RealDataLoader, ReferenceStore, filter_sessions_to_vocab,
        )
        REF_STORE_PATH = OUTPUT_DIR / f"reference_store_v{VOCAB_K}.joblib"
        if not REF_STORE_PATH.exists():
            raise RuntimeError(
                f"reference store not found at {REF_STORE_PATH}. "
                f"Run evaluate.py once to build it before enabling the in-train TSTR probe."
            )
        sampler_path = MODEL_DIR / "identity_sampler.pkl"
        if not sampler_path.exists():
            raise RuntimeError(
                f"identity sampler not found at {sampler_path}. "
                f"Run evaluate.py once to fit it before enabling the in-train TSTR probe."
            )

        print("\nLoading real val split for in-train TSTR probe ...")
        _real_data = RealDataLoader.load(
            max_train_sessions = 0,
            max_val_sessions   = None,
            include_test       = False,
        )
        val_real_split = _real_data.val_split
        del _real_data
        vocab_skus = set(get_sku2idx().keys())
        filter_sessions_to_vocab(val_real_split, vocab_skus)
        probe_ref_store             = ReferenceStore.load(REF_STORE_PATH)
        probe_identity_sampler_path = str(sampler_path)
        print(
            f"  TSTR probe ON: every {IS_BEST_EVAL_EVERY_N_EPOCHS} epochs, "
            f"{IS_BEST_EVAL_N_SESSIONS:,} sessions x {len(IS_BEST_EVAL_SEEDS)} seeds, "
            f"HR@{IS_BEST_EVAL_K} on {len(val_real_split):,} val sessions"
        )

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        epoch_action = epoch_item = epoch_temporal = epoch_category = 0.0
        epoch_grad_norm = 0.0
        epoch_batches = 0
        t0 = time.time()

        train_bar = tqdm(gen.loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS} train", leave=False)
        for batch in train_bar:
            events     = batch["events"].to(device)               # [B, T]
            items      = batch["items"].to(device)                # [B, T]
            deltas     = batch["deltas"].to(device)               # [B, T]
            categories = batch["categories"].to(device)           # [B, T]
            pad_mask   = batch["tgt_key_padding_mask"].to(device) # [B, T-1]
            lengths    = batch["lengths"]                         # [B]
            history    = batch.get("history")                     # dict or None
            if history is not None:
                history = {k: v.to(device) for k, v in history.items()}

            B, T = events.shape
            T_out = T - 1

            # Targets
            tgt_actions    = events[:, 1:]       # [B, T-1]
            tgt_items      = items[:, 1:]        # [B, T-1]
            tgt_deltas     = deltas[:, 1:]       # [B, T-1]
            tgt_categories = categories[:, 1:]  # [B, T-1]

            # Valid position mask (real, non-padded targets)
            valid = build_target_mask(lengths, T_out, device)  # [B, T-1]

            # Item mask: valid positions where target is an item-bearing event
            # Exclude OOV items (sku_idx=0): their sku_cat is PAD(0) which never
            # matches a real target category, producing -inf logits and nan loss.
            item_valid = valid & torch.isin(tgt_actions, item_action_ids) & (tgt_items > 0)

            with autocast("cuda", dtype=torch.bfloat16):
                action_logits, category_logits, temporal_logits, item_ctx = model(
                    events, items, deltas, categories,
                    history=history,
                    tgt_key_padding_mask=pad_mask,
                    item_mask=item_valid,
                )

                action_loss   = F.cross_entropy(action_logits[valid], tgt_actions[valid])
                temporal_loss = F.cross_entropy(temporal_logits[valid], tgt_deltas[valid])

                if item_valid.any():
                    h_item, tgt_e_item, tgt_c_item = item_ctx
                    cat_logits_for_loss = (
                        category_logits + cat_logit_adj
                        if cat_logit_adj is not None else category_logits
                    )
                    category_loss = F.cross_entropy(cat_logits_for_loss, tgt_categories[item_valid])
                    item_targets = tgt_items[item_valid]
                    sample_w = (
                        item_freq_weights[item_targets]
                        if item_freq_weights is not None else None
                    )
                    item_loss = item_head.loss(
                        h_item, tgt_e_item, tgt_c_item,
                        item_targets,
                        sample_weights=sample_w,
                    )
                else:
                    category_loss = torch.zeros((), device=device)
                    item_loss     = torch.zeros((), device=device)

                # Per-head loss weighting. Index map (matches LOSS_WEIGHTS):
                #   [0] action  [1] category  [2] item  [3] temporal
                if LOSS_WEIGHTS_ENABLED:
                    loss = (
                        LOSS_WEIGHTS[0] * action_loss
                        + LOSS_WEIGHTS[1] * category_loss
                        + LOSS_WEIGHTS[2] * item_loss
                        + LOSS_WEIGHTS[3] * temporal_loss
                    )
                else:
                    loss = action_loss + category_loss + item_loss + temporal_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()

            epoch_action    += action_loss.item()
            epoch_category  += category_loss.item()
            epoch_item      += item_loss.item()
            epoch_temporal  += temporal_loss.item()
            epoch_grad_norm += grad_norm
            epoch_batches   += 1

            train_bar.set_postfix(
                act=f"{epoch_action/epoch_batches:.3f}",
                cat=f"{epoch_category/epoch_batches:.3f}",
                item=f"{epoch_item/epoch_batches:.3f}",
                temp=f"{epoch_temporal/epoch_batches:.3f}",
            )

        elapsed      = time.time() - t0
        avg_action   = epoch_action   / epoch_batches
        avg_category = epoch_category / epoch_batches
        avg_item     = epoch_item     / epoch_batches
        avg_temporal = epoch_temporal / epoch_batches
        # Mirror the gradient's loss formula so the printed/logged total
        # matches what the optimizer actually sees.
        if LOSS_WEIGHTS_ENABLED:
            avg_loss = (
                LOSS_WEIGHTS[0] * avg_action
                + LOSS_WEIGHTS[1] * avg_category
                + LOSS_WEIGHTS[2] * avg_item
                + LOSS_WEIGHTS[3] * avg_temporal
            )
        else:
            avg_loss = avg_action + avg_category + avg_item + avg_temporal
        print(
            f"\nEpoch {epoch}/{TRAIN_EPOCHS} | "
            f"action={avg_action:.4f} cat={avg_category:.4f} "
            f"item={avg_item:.4f} temporal={avg_temporal:.4f} "
            f"| total={avg_loss:.4f} | {elapsed:.1f}s"
        )

        # Validation
        model.eval()
        val_action = val_category = val_item = val_temporal = 0.0
        val_batches = 0
        with torch.no_grad():
            val_bar = tqdm(val_gen.loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS} val  ", leave=False)
            for batch in val_bar:
                events     = batch["events"].to(device)
                items      = batch["items"].to(device)
                deltas     = batch["deltas"].to(device)
                categories = batch["categories"].to(device)
                pad_mask   = batch["tgt_key_padding_mask"].to(device)
                lengths    = batch["lengths"]
                history    = batch.get("history")
                if history is not None:
                    history = {k: v.to(device) for k, v in history.items()}

                B, T  = events.shape
                T_out = T - 1
                tgt_actions    = events[:, 1:]
                tgt_items      = items[:, 1:]
                tgt_deltas     = deltas[:, 1:]
                tgt_categories = categories[:, 1:]
                valid          = build_target_mask(lengths, T_out, device)

                item_valid = valid & torch.isin(tgt_actions, item_action_ids) & (tgt_items > 0)

                with autocast("cuda", dtype=torch.bfloat16):
                    action_logits, category_logits, temporal_logits, val_item_ctx = model(
                        events, items, deltas, categories,
                        history=history,
                        tgt_key_padding_mask=pad_mask,
                        item_mask=item_valid,
                    )
                    val_action_loss   = F.cross_entropy(action_logits[valid], tgt_actions[valid])
                    val_temporal_loss = F.cross_entropy(temporal_logits[valid], tgt_deltas[valid])

                    if item_valid.any():
                        h_item, tgt_e_item, tgt_c_item = val_item_ctx
                        val_cat_logits = (
                            category_logits + cat_logit_adj
                            if cat_logit_adj is not None else category_logits
                        )
                        val_cat_loss  = F.cross_entropy(val_cat_logits, tgt_categories[item_valid])
                        val_item_targets = tgt_items[item_valid]
                        val_sample_w = (
                            item_freq_weights[val_item_targets]
                            if item_freq_weights is not None else None
                        )
                        val_item_loss = item_head.loss(
                            h_item, tgt_e_item, tgt_c_item,
                            val_item_targets,
                            sample_weights=val_sample_w,
                        )
                    else:
                        val_cat_loss  = torch.zeros((), device=device)
                        val_item_loss = torch.zeros((), device=device)

                val_action   += val_action_loss.item()
                val_category += val_cat_loss.item()
                val_item     += val_item_loss.item()
                val_temporal += val_temporal_loss.item()
                val_batches  += 1

        val_avg_action   = val_action   / val_batches
        val_avg_category = val_category / val_batches
        val_avg_item     = val_item     / val_batches
        val_avg_temporal = val_temporal / val_batches
        if LOSS_WEIGHTS_ENABLED:
            val_loss = (
                LOSS_WEIGHTS[0] * val_avg_action
                + LOSS_WEIGHTS[1] * val_avg_category
                + LOSS_WEIGHTS[2] * val_avg_item
                + LOSS_WEIGHTS[3] * val_avg_temporal
            )
        else:
            val_loss = val_avg_action + val_avg_category + val_avg_item + val_avg_temporal
        print(
            f"Val   {epoch}/{TRAIN_EPOCHS} | "
            f"action={val_avg_action:.4f} "
            f"cat={val_avg_category:.4f} "
            f"item={val_avg_item:.4f} "
            f"temporal={val_avg_temporal:.4f} "
            f"| total={val_loss:.4f}"
        )

        # Diagnostic logging — always log val/total even when the TSTR probe
        # drives is_best, so per-head dynamics are still visible in wandb.
        log_epoch = {
            "epoch": epoch,
            "train/action_loss":   avg_action,
            "train/category_loss": avg_category,
            "train/item_loss":     avg_item,
            "train/temporal_loss": avg_temporal,
            "train/total_loss":    avg_loss,
            "train/lr":            scheduler.get_last_lr()[0],
            "train/grad_norm":     epoch_grad_norm / epoch_batches,
            "train/time_s":        elapsed,
            "val/action_loss":     val_avg_action,
            "val/category_loss":   val_avg_category,
            "val/item_loss":       val_avg_item,
            "val/temporal_loss":   val_avg_temporal,
            "val/total_loss":      val_loss,
        }

        is_probe_epoch = (
            IS_BEST_EVAL_ENABLED
            and (epoch % IS_BEST_EVAL_EVERY_N_EPOCHS == 0 or epoch == TRAIN_EPOCHS)
        )
        is_best = False

        if is_probe_epoch:
            # TSTR-T probe: candidate -> generate -> GRU4Rec on synth -> HR@K on val.
            # Mean across IS_BEST_EVAL_SEEDS replaces val_loss as the is_best signal.
            from evaluation.in_train_tstr import run_tstr_probe

            candidate_path = run_subdir / "model_candidate.pt"
            model.save(candidate_path)

            # Free GPU for generation + GRU4Rec; transformer goes to CPU.
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            print(
                f"\n  >> TSTR probe @ epoch {epoch}: "
                f"{IS_BEST_EVAL_N_SESSIONS:,} sessions x {len(IS_BEST_EVAL_SEEDS)} seeds"
            )
            try:
                probe = run_tstr_probe(
                    candidate_path        = candidate_path,
                    val_split             = val_real_split,
                    n_sessions            = IS_BEST_EVAL_N_SESSIONS,
                    seeds                 = list(IS_BEST_EVAL_SEEDS),
                    ref_store             = probe_ref_store,
                    identity_sampler_path = probe_identity_sampler_path,
                    k                     = IS_BEST_EVAL_K,
                    device                = device,
                )
            finally:
                # Always restore training model, even if probe blew up.
                model.to(device)
                gc.collect()
                torch.cuda.empty_cache()

            log_epoch.update({
                f"tstr/hr_at_{IS_BEST_EVAL_K}_mean":   probe["hr_mean"],
                f"tstr/hr_at_{IS_BEST_EVAL_K}_std":    probe["hr_std"],
                f"tstr/ndcg_at_{IS_BEST_EVAL_K}_mean": probe["ndcg_mean"],
                f"tstr/ndcg_at_{IS_BEST_EVAL_K}_std":  probe["ndcg_std"],
            })
            print(
                f"  TSTR probe @ epoch {epoch}: "
                f"HR@{IS_BEST_EVAL_K}={probe['hr_mean']:.4f} ± {probe['hr_std']:.4f}  "
                f"NDCG@{IS_BEST_EVAL_K}={probe['ndcg_mean']:.4f} ± {probe['ndcg_std']:.4f}  "
                f"per-seed={probe['hr_per_seed']}"
            )

            is_best = probe["hr_mean"] > best_hr
            if is_best:
                best_hr = probe["hr_mean"]
                probes_no_improve = 0
                target = run_subdir / "model.pt"
                target.unlink(missing_ok=True)
                candidate_path.rename(target)
                cfg_path = run_subdir / "model.yaml"
                snapshot = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
                snapshot["_checkpoint"] = {
                    "tstr_hr_mean":  round(float(best_hr), 6),
                    "tstr_hr_std":   round(float(probe["hr_std"]), 6),
                    "tstr_ndcg_mean": round(float(probe["ndcg_mean"]), 6),
                    "val_loss":      round(float(val_loss), 6),
                    "epoch":         epoch,
                }
                with open(cfg_path, "w") as f:
                    yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
                print(
                    f"  Checkpoint promoted -> {target.name}  "
                    f"(TSTR HR@{IS_BEST_EVAL_K}={best_hr:.4f} ± {probe['hr_std']:.4f})"
                )
                wandb.summary["best_tstr_hr"]   = best_hr
                wandb.summary["best_tstr_std"]  = probe["hr_std"]
                wandb.summary["best_epoch"]     = epoch
            else:
                candidate_path.unlink(missing_ok=True)
                probes_no_improve += 1
                print(
                    f"  Probe did not improve "
                    f"(current HR={probe['hr_mean']:.4f}, best={best_hr:.4f}, "
                    f"probes_no_improve={probes_no_improve})"
                )

        elif not IS_BEST_EVAL_ENABLED:
            # Legacy val_loss-based is_best (probe disabled).
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                epochs_no_improve = 0
                ckpt_path = run_subdir / "model.pt"
                cfg_path  = run_subdir / "model.yaml"
                model.save(ckpt_path)
                snapshot = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
                snapshot["_checkpoint"] = {
                    "val_loss": round(float(best_loss), 6),
                    "epoch":    epoch,
                }
                with open(cfg_path, "w") as f:
                    yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
                print(f"  Checkpoint saved -> {ckpt_path.name}  (val_loss={best_loss:.4f})")
                wandb.summary["best_val_loss"] = best_loss
                wandb.summary["best_epoch"]    = epoch
            else:
                epochs_no_improve += 1

        log_epoch["val/is_best"] = int(is_best)
        wandb.log(log_epoch)

        # Early-stopping: probe-mode counts probes-without-improvement (so
        # patience=N means N consecutive non-improving probes, i.e. roughly
        # N * IS_BEST_EVAL_EVERY_N_EPOCHS epochs of stagnation).
        if TRAIN_PATIENCE is not None:
            if IS_BEST_EVAL_ENABLED and probes_no_improve >= TRAIN_PATIENCE:
                print(
                    f"  Early stopping: no TSTR HR improvement for "
                    f"{probes_no_improve} probe(s) (patience={TRAIN_PATIENCE})."
                )
                wandb.summary["early_stopped"]       = True
                wandb.summary["early_stopped_epoch"] = epoch
                break
            if (not IS_BEST_EVAL_ENABLED) and epochs_no_improve >= TRAIN_PATIENCE:
                print(
                    f"  Early stopping: no val_loss improvement for "
                    f"{epochs_no_improve} epoch(s) (patience={TRAIN_PATIENCE})."
                )
                wandb.summary["early_stopped"]       = True
                wandb.summary["early_stopped_epoch"] = epoch
                break

    if IS_BEST_EVAL_ENABLED:
        print(f"\nTraining complete. Best TSTR HR@{IS_BEST_EVAL_K}: {best_hr:.4f}")
    else:
        print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Model saved to: {run_subdir / 'model.pt'}")
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--comment", default=None,
        help="Suffix appended to the run folder name for easier identification.",
    )
    args = p.parse_args()
    comment = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", args.comment.strip()).strip("_")
        if args.comment else None
    )
    train(comment=comment)
