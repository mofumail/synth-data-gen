"""
SessionTransformer training script.

Three factored cross-entropy losses:
    action_loss   : CE over all real positions
    item_loss     : CE restricted to item-bearing target positions
    temporal_loss : CE over all real positions

All hyperparameters are read from config.yaml.

Usage:
    PYTHONPATH=. uv run python train.py

Checkpoint saved to MODEL_DIR/session_transformer.pt on val loss improvement.
A config.yaml snapshot is written to MODEL_DIR alongside the checkpoint.
"""

import time
from pathlib import Path

import yaml
import wandb
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    MODEL_DIR, MODEL_NAME, MODEL_SUBDIR, VOCAB_K, N_TEMPORAL_BINS, HISTORY_WINDOW,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH,
    TRAIN_LR, TRAIN_D_MODEL, TRAIN_N_LAYERS, TRAIN_N_HEADS,
    TRAIN_MAX_SESSIONS, TRAIN_NUM_WORKERS,
    TRAIN_ITEM_LOSS, TRAIN_SAMPLED_K, TRAIN_SAMPLED_ALPHA,
    N_CATEGORIES, CAT2IDX_PATH,
)
from ingestion.dataset import (
    InteractionGenerator, ensure_vocab_stats,
    get_cat_vocab, get_sku2idx, get_sku_properties,
)
from simulation.generator.session_transformer import (
    SessionTransformer,
    N_ACTIONS,
    ITEM_BEARING_IDX,
)


def build_target_mask(lengths: torch.Tensor, T_out: int, device) -> torch.Tensor:
    """
    True at positions that are REAL targets (not padding).
    For session of length L, real output positions are 0..L-2
    (predicting tokens 1..L-1 from inputs 0..L-2).
    """
    positions = torch.arange(T_out, device=device).unsqueeze(0)   # [1, T_out]
    return positions < (lengths.to(device).unsqueeze(1) - 1)       # [B, T_out]


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: epochs={TRAIN_EPOCHS}, batch={TRAIN_BATCH_SIZE}, "
          f"d_model={TRAIN_D_MODEL}, layers={TRAIN_N_LAYERS}, heads={TRAIN_N_HEADS}, "
          f"lr={TRAIN_LR}, max_sessions={TRAIN_MAX_SESSIONS}")
    print(f"Item loss: {TRAIN_ITEM_LOSS}"
          + (f" (K={TRAIN_SAMPLED_K}, alpha={TRAIN_SAMPLED_ALPHA})"
             if TRAIN_ITEM_LOSS in ("sampled", "hierarchical") else "")
          + (f" | n_categories={N_CATEGORIES}"
             if TRAIN_ITEM_LOSS == "hierarchical" else ""))

    run = wandb.init(
        project="thesis-session-transformer",
        name=MODEL_NAME,
        config={
            "model":           MODEL_NAME,
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
            "item_loss":       TRAIN_ITEM_LOSS,
            "sampled_k":       TRAIN_SAMPLED_K,
            "sampled_alpha":   TRAIN_SAMPLED_ALPHA,
            "n_categories":    N_CATEGORIES if TRAIN_ITEM_LOSS == "hierarchical" else 0,
        },
    )

    # Epoch-level metrics use epoch as x-axis; step-level metrics use global_step
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch_*", step_metric="epoch")
    wandb.define_metric("val/*",         step_metric="epoch")

    # Dataset
    print(f"\nLoading dataset (split=train, max_sessions={TRAIN_MAX_SESSIONS}) ...")
    gen = InteractionGenerator(
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=TRAIN_NUM_WORKERS,
        max_sessions=TRAIN_MAX_SESSIONS,
    )
    print(f"  {len(gen.dataset):,} training sessions | {len(gen):,} batches/epoch")

    print(f"\nLoading validation dataset ...")
    val_gen = InteractionGenerator(
        split="val",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=TRAIN_NUM_WORKERS,
        max_sessions=None,        # always use full val set
    )
    print(f"  {len(val_gen.dataset):,} val sessions | {len(val_gen):,} batches")

    # Category vocab (hierarchical mode): load cat_sku_pools as GPU tensors.
    # These are used to build the masked softmax so the item-head loss normalizes
    # only over SKUs in the target category (the whole point of the factorization).
    cat_sku_pools_tensors = None
    if TRAIN_ITEM_LOSS == "hierarchical":
        if not CAT2IDX_PATH.exists():
            raise RuntimeError(
                "cat2idx.joblib not found. Run preprocess.py first to build category vocab."
            )
        sku2idx = get_sku2idx()
        _, cat_pools_np = get_cat_vocab(sku2idx)   # list[np.ndarray], indexed by dense cat_idx
        cat_sku_pools_tensors = [
            torch.as_tensor(p, dtype=torch.long, device=device) for p in cat_pools_np
        ]
        nonempty = sum(1 for p in cat_sku_pools_tensors if p.numel() > 0)
        avg_pool = sum(p.numel() for p in cat_sku_pools_tensors) / max(nonempty, 1)
        print(f"  Hierarchical mode: {N_CATEGORIES} categories "
              f"({nonempty} non-empty pools, avg pool size {avg_pool:.0f})")

    # SKU property tables (price + quantized name) — used as input-side features
    # in the encoder. Looked up by SKU index inside the model.
    sku_props = None
    if TRAIN_ITEM_LOSS == "hierarchical":
        _sku2idx_for_props = get_sku2idx()
        sku_props = get_sku_properties(_sku2idx_for_props)
        print(f"  SKU aux features: price + name (built/loaded from sku_properties.joblib)")

    # Model
    model = SessionTransformer(
        vocab_size=VOCAB_K,
        d_model=TRAIN_D_MODEL,
        n_layers=TRAIN_N_LAYERS,
        n_heads=TRAIN_N_HEADS,
        n_temporal_bins=N_TEMPORAL_BINS,
        window_H=HISTORY_WINDOW,
        valid_transitions={},     # mask not used during training
        n_categories=N_CATEGORIES if TRAIN_ITEM_LOSS == "hierarchical" else 0,
        sku_price_table=sku_props["price"] if sku_props else None,
        sku_name_table =sku_props["name"]  if sku_props else None,
    ).to(device)
    # Capture the uncompiled item_head BEFORE torch.compile so sampled_logits()
    # reaches the real module rather than going through the OptimizedModule wrapper.
    item_head = model.item_head
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Sampled-softmax sampler (only for "sampled" mode).
    if TRAIN_ITEM_LOSS == "sampled":
        counts_np = ensure_vocab_stats()                       # [V] int64
        counts = torch.from_numpy(counts_np).to(device)
        probs  = counts.double().pow(TRAIN_SAMPLED_ALPHA)
        probs[0] = 0.0                                         # padding: never drawn
        probs  = (probs / probs.sum()).float()                 # [V] fp32, ~2.5 MB
        log_q  = torch.log(probs.clamp_min(1e-30))             # [V] fp32
        nonzero = int((probs > 0).sum().item())
        print(f"  Sampled softmax: V={counts.numel():,}, "
              f"active={nonzero:,}, K={TRAIN_SAMPLED_K}, alpha={TRAIN_SAMPLED_ALPHA}")
    else:
        probs = log_q = None

    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS * len(gen), eta_min=1e-6)
    scaler    = GradScaler("cuda")

    MODEL_SUBDIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        epoch_action = epoch_item = epoch_temporal = epoch_category = 0.0
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
            tgt_is_item = torch.zeros_like(tgt_actions, dtype=torch.bool)
            for idx in ITEM_BEARING_IDX:
                tgt_is_item |= (tgt_actions == idx)
            item_valid = valid & tgt_is_item  # [B, T-1]

            # Forward (teacher-forced).
            # sampled/hierarchical use "ctx" to get (h_item, tgt_e_item, tgt_c_item)
            # back and skip the full [M, V] projection; full mode keeps it.
            _fwd_mode = "ctx" if TRAIN_ITEM_LOSS in ("sampled", "hierarchical") else "full"

            with autocast("cuda"):
                action_logits, category_logits, item_logits, temporal_logits, item_ctx = model(
                    events, items, deltas, categories,
                    history=history,
                    tgt_key_padding_mask=pad_mask,
                    item_mask=item_valid,
                    item_head_mode=_fwd_mode,
                )

                # Action loss
                a_logits    = action_logits[valid]          # [N, N_ACTIONS]
                a_tgt       = tgt_actions[valid]            # [N]
                action_loss = F.cross_entropy(a_logits, a_tgt)

                # Category loss (hierarchical only)
                category_loss = torch.zeros(1, device=device).squeeze()
                if TRAIN_ITEM_LOSS == "hierarchical" and item_valid.any() and category_logits is not None:
                    c_tgt         = tgt_categories[item_valid]   # [M]
                    category_loss = F.cross_entropy(category_logits, c_tgt)

                # Item loss
                if item_valid.any():
                    i_tgt = tgt_items[item_valid]            # [M]
                    if TRAIN_ITEM_LOSS == "sampled":
                        h_item, tgt_e_item, _ = item_ctx
                        neg_ids = torch.multinomial(probs, TRAIN_SAMPLED_K, replacement=True)
                        sampled = item_head.sampled_logits(h_item, tgt_e_item, i_tgt, neg_ids, log_q)
                        zeros   = torch.zeros(sampled.size(0), dtype=torch.long, device=device)
                        item_loss = F.cross_entropy(sampled, zeros)
                    elif TRAIN_ITEM_LOSS == "hierarchical":
                        # Within-category CE computed directly from the conditioning
                        # vector: never materializes the full [M, V] projection.
                        # cond = h + action_emb + category_emb; per category slice
                        # fc.weight[pool] and project only onto pool columns.
                        h_item, tgt_e_item, tgt_c_item = item_ctx
                        cond = (
                            h_item
                            + item_head.action_cond(tgt_e_item)
                            + item_head.category_cond(tgt_c_item)
                        )                                                       # [M, d]
                        W = item_head.fc.weight                                  # [V, d]
                        b = item_head.fc.bias                                    # [V]
                        loss_sum = torch.zeros((), device=device)
                        total_m  = 0
                        for cat_idx in torch.unique(tgt_c_item).tolist():
                            row_ids = (tgt_c_item == cat_idx).nonzero(as_tuple=False).squeeze(-1)
                            pool    = cat_sku_pools_tensors[cat_idx]
                            if pool.numel() == 0:
                                continue
                            w_pool   = W.index_select(0, pool)                  # [K, d]
                            b_pool   = b.index_select(0, pool)                  # [K]
                            sub_logits = cond[row_ids] @ w_pool.t() + b_pool     # [R, K]
                            local_tgt  = torch.searchsorted(pool, i_tgt[row_ids])
                            local_tgt  = local_tgt.clamp(max=pool.numel() - 1)
                            loss_sum  = loss_sum + F.cross_entropy(
                                sub_logits, local_tgt, reduction="sum"
                            )
                            total_m  += row_ids.numel()
                        item_loss = loss_sum / max(total_m, 1)
                    else:
                        item_loss = F.cross_entropy(item_logits, i_tgt)
                else:
                    item_loss = torch.zeros(1, device=device).squeeze()

                # Temporal loss
                d_logits      = temporal_logits[valid]       # [N, N_BINS]
                d_tgt         = tgt_deltas[valid]            # [N]
                temporal_loss = F.cross_entropy(d_logits, d_tgt)

                loss = action_loss + category_loss + item_loss + temporal_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_action   += action_loss.item()
            epoch_category += category_loss.item()
            epoch_item     += item_loss.item()
            epoch_temporal += temporal_loss.item()
            epoch_batches  += 1
            global_step    += 1

            pf = dict(
                act=f"{epoch_action/epoch_batches:.3f}",
                item=f"{epoch_item/epoch_batches:.3f}",
                temp=f"{epoch_temporal/epoch_batches:.3f}",
            )
            if TRAIN_ITEM_LOSS == "hierarchical":
                pf["cat"] = f"{epoch_category/epoch_batches:.3f}"
            train_bar.set_postfix(**pf)

            # Log every 100 steps to keep W&B traffic low
            if global_step % 100 == 0:
                log_dict = {
                    "train/action_loss":   action_loss.item(),
                    "train/item_loss":     item_loss.item(),
                    "train/temporal_loss": temporal_loss.item(),
                    "train/total_loss":    loss.item(),
                    "train/lr":            scheduler.get_last_lr()[0],
                    "train/grad_norm":     grad_norm,
                    "epoch": epoch,
                }
                if TRAIN_ITEM_LOSS == "hierarchical":
                    log_dict["train/category_loss"] = category_loss.item()
                wandb.log(log_dict, step=global_step)

        elapsed      = time.time() - t0
        avg_action   = epoch_action   / epoch_batches
        avg_category = epoch_category / epoch_batches
        avg_item     = epoch_item     / epoch_batches
        avg_temporal = epoch_temporal / epoch_batches
        avg_loss     = avg_action + avg_category + avg_item + avg_temporal
        cat_str = f"cat={avg_category:.4f} " if TRAIN_ITEM_LOSS == "hierarchical" else ""
        print(
            f"\nEpoch {epoch}/{TRAIN_EPOCHS} | "
            f"action={avg_action:.4f} "
            f"{cat_str}"
            f"item={avg_item:.4f} "
            f"temporal={avg_temporal:.4f} "
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

                tgt_is_item = torch.zeros_like(tgt_actions, dtype=torch.bool)
                for idx in ITEM_BEARING_IDX:
                    tgt_is_item |= (tgt_actions == idx)
                item_valid = valid & tgt_is_item

                with autocast("cuda"):
                    # Val always uses full CE for item loss (apples-to-apples comparison).
                    action_logits, category_logits, item_logits, temporal_logits, _ = model(
                        events, items, deltas, categories,
                        history=history,
                        tgt_key_padding_mask=pad_mask,
                        item_mask=item_valid,
                        item_head_mode="full",
                    )
                    a_logits        = action_logits[valid]
                    val_action_loss = F.cross_entropy(a_logits, tgt_actions[valid])

                    val_cat_loss = torch.zeros(1, device=device).squeeze()
                    if TRAIN_ITEM_LOSS == "hierarchical" and item_valid.any() and category_logits is not None:
                        val_cat_loss = F.cross_entropy(category_logits, tgt_categories[item_valid])

                    if item_valid.any():
                        val_item_loss = F.cross_entropy(item_logits, tgt_items[item_valid])
                    else:
                        val_item_loss = torch.zeros(1, device=device).squeeze()

                    d_logits          = temporal_logits[valid]
                    val_temporal_loss = F.cross_entropy(d_logits, tgt_deltas[valid])

                val_action   += val_action_loss.item()
                val_category += val_cat_loss.item()
                val_item     += val_item_loss.item()
                val_temporal += val_temporal_loss.item()
                val_batches  += 1

        val_avg_action   = val_action   / val_batches
        val_avg_category = val_category / val_batches
        val_avg_item     = val_item     / val_batches
        val_avg_temporal = val_temporal / val_batches
        val_loss         = val_avg_action + val_avg_category + val_avg_item + val_avg_temporal
        val_cat_str = f"cat={val_avg_category:.4f} " if TRAIN_ITEM_LOSS == "hierarchical" else ""
        print(
            f"Val   {epoch}/{TRAIN_EPOCHS} | "
            f"action={val_avg_action:.4f} "
            f"{val_cat_str}"
            f"item={val_avg_item:.4f} "
            f"temporal={val_avg_temporal:.4f} "
            f"| total={val_loss:.4f}"
        )

        is_best  = val_loss < best_loss
        log_epoch = {
            "epoch": epoch,
            # Epoch-level train averages
            "train/epoch_action_loss":   avg_action,
            "train/epoch_item_loss":     avg_item,
            "train/epoch_temporal_loss": avg_temporal,
            "train/epoch_total_loss":    avg_loss,
            "train/epoch_time_s":        elapsed,
            # Val
            "val/action_loss":           val_avg_action,
            "val/item_loss":             val_avg_item,
            "val/temporal_loss":         val_avg_temporal,
            "val/total_loss":            val_loss,
            "val/is_best":               int(is_best),
        }
        if TRAIN_ITEM_LOSS == "hierarchical":
            log_epoch["train/epoch_category_loss"] = avg_category
            log_epoch["val/category_loss"]         = val_avg_category
        wandb.log(log_epoch, step=global_step)

        if is_best:
            best_loss = val_loss
            ckpt_path = MODEL_SUBDIR / "model.pt"
            cfg_path  = MODEL_SUBDIR / "model.yaml"
            model.save(ckpt_path)
            # Save config snapshot alongside checkpoint for reproducibility
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

    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Model saved to: {MODEL_SUBDIR / 'model.pt'}")
    wandb.finish()


if __name__ == "__main__":
    train()
