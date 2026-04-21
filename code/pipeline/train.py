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
    TRAIN_MAX_SESSIONS, TRAIN_NUM_WORKERS,
    N_CATEGORIES, CAT2IDX_PATH,
    SVDPQ_ENABLED, SVDPQ_T, SVDPQ_V,
)
from ingestion.dataset import (
    InteractionGenerator, get_cat_vocab, get_sku2idx, get_sku_properties,
    get_sku_tokens,
)
from simulation.generator.session_transformer import (
    SessionTransformer, SVDPQItemHead,
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
    # Stamp the run once at start so every epoch writes into the same folder.
    run_stamp  = datetime.now().strftime("%d%m%y-%H-%M-%S")
    run_name   = f"{MODEL_NAME}_{run_stamp}"
    run_subdir = MODEL_DIR / run_name
    run_subdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Run:    {run_name}")
    print(f"Config: epochs={TRAIN_EPOCHS}, batch={TRAIN_BATCH_SIZE}, "
          f"d_model={TRAIN_D_MODEL}, layers={TRAIN_N_LAYERS}, heads={TRAIN_N_HEADS}, "
          f"lr={TRAIN_LR}, max_sessions={TRAIN_MAX_SESSIONS}, n_categories={N_CATEGORIES}")

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
        },
    )

    # All metrics should use epoch as x-axis
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*",   step_metric="epoch")

    # Dataset
    print(f"\nLoading dataset (split=train, max_sessions={TRAIN_MAX_SESSIONS}) ...")
    gen = InteractionGenerator(
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=TRAIN_NUM_WORKERS,
        max_sessions=TRAIN_MAX_SESSIONS,
    )
    print(f"{len(gen.dataset):,} training sessions | {len(gen):,} batches/epoch")

    print(f"\nLoading validation dataset ...")
    val_gen = InteractionGenerator(
        split="val",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=TRAIN_NUM_WORKERS,
        max_sessions=None,        # always use full val set
    )
    print(f"{len(val_gen.dataset):,} val sessions | {len(val_gen):,} batches")

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
    model = torch.compile(model)
    # model = torch.compile(model, dymamic=True) #

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS * len(gen), eta_min=1e-6)

    best_loss = float("inf")

    # Precomputed item-bearing action ids for torch.isin masking
    item_action_ids = torch.tensor(sorted(ITEM_BEARING_IDX), dtype=torch.long, device=device)

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
                    category_loss = F.cross_entropy(category_logits, tgt_categories[item_valid])
                    item_loss = item_head.loss(
                        h_item, tgt_e_item, tgt_c_item,
                        tgt_items[item_valid],
                    )
                else:
                    category_loss = torch.zeros((), device=device)
                    item_loss     = torch.zeros((), device=device)

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
        avg_loss     = avg_action + avg_category + avg_item + avg_temporal
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
                        val_cat_loss  = F.cross_entropy(category_logits, tgt_categories[item_valid])
                        val_item_loss = item_head.loss(
                            h_item, tgt_e_item, tgt_c_item,
                            tgt_items[item_valid],
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
        val_loss         = val_avg_action + val_avg_category + val_avg_item + val_avg_temporal
        print(
            f"Val   {epoch}/{TRAIN_EPOCHS} | "
            f"action={val_avg_action:.4f} "
            f"cat={val_avg_category:.4f} "
            f"item={val_avg_item:.4f} "
            f"temporal={val_avg_temporal:.4f} "
            f"| total={val_loss:.4f}"
        )

        is_best  = val_loss < best_loss
        log_epoch = {
            "epoch": epoch,
            # Train
            "train/action_loss":   avg_action,
            "train/category_loss": avg_category,
            "train/item_loss":     avg_item,
            "train/temporal_loss": avg_temporal,
            "train/total_loss":    avg_loss,
            "train/lr":            scheduler.get_last_lr()[0],
            "train/grad_norm":     epoch_grad_norm / epoch_batches,
            "train/time_s":        elapsed,
            # Val
            "val/action_loss":     val_avg_action,
            "val/category_loss":   val_avg_category,
            "val/item_loss":       val_avg_item,
            "val/temporal_loss":   val_avg_temporal,
            "val/total_loss":      val_loss,
            "val/is_best":         int(is_best),
        }
        wandb.log(log_epoch)

        if is_best:
            best_loss = val_loss
            ckpt_path = run_subdir / "model.pt"
            cfg_path  = run_subdir / "model.yaml"
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
    print(f"Model saved to: {run_subdir / 'model.pt'}")
    wandb.finish()


if __name__ == "__main__":
    train()
