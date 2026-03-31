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
    TRAIN_MAX_SESSIONS,
)
from ingestion.dataset import InteractionGenerator
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
        },
    )

    # Dataset
    print(f"\nLoading dataset (split=train, max_sessions={TRAIN_MAX_SESSIONS}) ...")
    gen = InteractionGenerator(
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=0,            # safer for large parquet + multiprocessing
        max_sessions=TRAIN_MAX_SESSIONS,
    )
    print(f"  {len(gen.dataset):,} training sessions | {len(gen):,} batches/epoch")

    print(f"\nLoading validation dataset ...")
    val_gen = InteractionGenerator(
        split="val",
        batch_size=TRAIN_BATCH_SIZE,
        max_length=TRAIN_MAX_LENGTH,
        num_workers=0,
        max_sessions=None,        # always use full val set
    )
    print(f"  {len(val_gen.dataset):,} val sessions | {len(val_gen):,} batches")

    # Model
    model = SessionTransformer(
        vocab_size=VOCAB_K,
        d_model=TRAIN_D_MODEL,
        n_layers=TRAIN_N_LAYERS,
        n_heads=TRAIN_N_HEADS,
        n_temporal_bins=N_TEMPORAL_BINS,
        window_H=HISTORY_WINDOW,
        valid_transitions={},     # mask not used during training
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS * len(gen), eta_min=1e-6)
    scaler    = GradScaler("cuda")

    MODEL_SUBDIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        epoch_action = epoch_item = epoch_temporal = 0.0
        epoch_batches = 0
        t0 = time.time()

        train_bar = tqdm(gen.loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS} train", leave=False)
        for batch in train_bar:
            events   = batch["events"].to(device)               # [B, T]
            items    = batch["items"].to(device)                # [B, T]
            deltas   = batch["deltas"].to(device)               # [B, T]
            pad_mask = batch["tgt_key_padding_mask"].to(device) # [B, T-1]
            lengths  = batch["lengths"]                         # [B]
            history  = batch.get("history")                     # dict or None
            if history is not None:
                history = {k: v.to(device) for k, v in history.items()}

            B, T = events.shape
            T_out = T - 1

            # Targets
            tgt_actions = events[:, 1:]   # [B, T-1]
            tgt_items   = items[:, 1:]    # [B, T-1]
            tgt_deltas  = deltas[:, 1:]   # [B, T-1]

            # Valid position mask (real, non-padded targets)
            valid = build_target_mask(lengths, T_out, device)  # [B, T-1]

            # Item mask: valid positions where target is an item-bearing event
            tgt_is_item = torch.zeros_like(tgt_actions, dtype=torch.bool)
            for idx in ITEM_BEARING_IDX:
                tgt_is_item |= (tgt_actions == idx)
            item_valid = valid & tgt_is_item  # [B, T-1]

            # Forward (teacher-forced); item_logits are sparse [M, vocab_size]
            with autocast("cuda"):
                action_logits, item_logits, temporal_logits = model(
                    events, items, deltas,
                    history=history,
                    tgt_key_padding_mask=pad_mask,
                    item_mask=item_valid,
                )
                # action_logits:   [B, T-1, n_actions]
                # item_logits:     [M, vocab_size]   M = item_valid.sum()
                # temporal_logits: [B, T-1, n_bins]

                # Action loss
                a_logits = action_logits[valid]          # [N, N_ACTIONS]
                a_tgt    = tgt_actions[valid]            # [N]
                action_loss = F.cross_entropy(a_logits, a_tgt)

                # Item loss (sparse logits already at item positions)
                if item_valid.any():
                    i_tgt     = tgt_items[item_valid]    # [M]
                    item_loss = F.cross_entropy(item_logits, i_tgt)
                else:
                    item_loss = torch.zeros(1, device=device).squeeze()

                # Temporal loss
                d_logits = temporal_logits[valid]        # [N, N_BINS]
                d_tgt    = tgt_deltas[valid]             # [N]
                temporal_loss = F.cross_entropy(d_logits, d_tgt)

                loss = action_loss + item_loss + temporal_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_action   += action_loss.item()
            epoch_item     += item_loss.item()
            epoch_temporal += temporal_loss.item()
            epoch_batches  += 1
            global_step    += 1

            train_bar.set_postfix(
                act=f"{epoch_action/epoch_batches:.3f}",
                item=f"{epoch_item/epoch_batches:.3f}",
                temp=f"{epoch_temporal/epoch_batches:.3f}",
            )

            # Log every 100 steps to keep W&B traffic low
            if global_step % 100 == 0:
                wandb.log({
                    "train/action_loss":   action_loss.item(),
                    "train/item_loss":     item_loss.item(),
                    "train/temporal_loss": temporal_loss.item(),
                    "train/total_loss":    loss.item(),
                    "train/lr":            scheduler.get_last_lr()[0],
                    "train/grad_norm":     grad_norm,
                    "epoch": epoch,
                }, step=global_step)

        elapsed  = time.time() - t0
        avg_action   = epoch_action   / epoch_batches
        avg_item     = epoch_item     / epoch_batches
        avg_temporal = epoch_temporal / epoch_batches
        avg_loss     = avg_action + avg_item + avg_temporal
        print(
            f"\nEpoch {epoch}/{TRAIN_EPOCHS} | "
            f"action={avg_action:.4f} "
            f"item={avg_item:.4f} "
            f"temporal={avg_temporal:.4f} "
            f"| total={avg_loss:.4f} | {elapsed:.1f}s"
        )

        # Validation
        model.eval()
        val_action = val_item = val_temporal = 0.0
        val_batches = 0
        with torch.no_grad():
            val_bar = tqdm(val_gen.loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS} val  ", leave=False)
            for batch in val_bar:
                events   = batch["events"].to(device)
                items    = batch["items"].to(device)
                deltas   = batch["deltas"].to(device)
                pad_mask = batch["tgt_key_padding_mask"].to(device)
                lengths  = batch["lengths"]
                history  = batch.get("history")
                if history is not None:
                    history = {k: v.to(device) for k, v in history.items()}

                B, T  = events.shape
                T_out = T - 1
                tgt_actions = events[:, 1:]
                tgt_items   = items[:, 1:]
                tgt_deltas  = deltas[:, 1:]
                valid       = build_target_mask(lengths, T_out, device)

                tgt_is_item = torch.zeros_like(tgt_actions, dtype=torch.bool)
                for idx in ITEM_BEARING_IDX:
                    tgt_is_item |= (tgt_actions == idx)
                item_valid = valid & tgt_is_item

                with autocast("cuda"):
                    action_logits, item_logits, temporal_logits = model(
                        events, items, deltas,
                        history=history,
                        tgt_key_padding_mask=pad_mask,
                        item_mask=item_valid,
                    )
                    a_logits = action_logits[valid]
                    val_action_loss = F.cross_entropy(a_logits, tgt_actions[valid])

                    if item_valid.any():
                        val_item_loss = F.cross_entropy(item_logits, tgt_items[item_valid])
                    else:
                        val_item_loss = torch.zeros(1, device=device).squeeze()

                    d_logits = temporal_logits[valid]
                    val_temporal_loss = F.cross_entropy(d_logits, tgt_deltas[valid])

                val_action   += val_action_loss.item()
                val_item     += val_item_loss.item()
                val_temporal += val_temporal_loss.item()
                val_batches  += 1

        val_avg_action   = val_action   / val_batches
        val_avg_item     = val_item     / val_batches
        val_avg_temporal = val_temporal / val_batches
        val_loss         = val_avg_action + val_avg_item + val_avg_temporal
        print(
            f"Val   {epoch}/{TRAIN_EPOCHS} | "
            f"action={val_avg_action:.4f} "
            f"item={val_avg_item:.4f} "
            f"temporal={val_avg_temporal:.4f} "
            f"| total={val_loss:.4f}"
        )

        is_best = val_loss < best_loss
        wandb.log({
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
        }, step=global_step)

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
