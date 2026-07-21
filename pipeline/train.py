"""
SessionTransformer training script.

Three factored cross-entropy losses:
    action_loss   : CE over all real positions
    item_loss     : mean CE over N_CODE_LEVELS RQ-VAE code levels,
                    restricted to item-bearing target positions
    temporal_loss : CE over all real positions

All hyperparameters are read from config.yaml.

Usage:
    PYTHONPATH=. uv run python train.py
    PYTHONPATH=. uv run python train.py --rebuild   # delete and rebuild RQ-VAE + CTGAN first

Checkpoint saved to MODEL_SUBDIR/model.pt on val loss improvement.
A config.yaml snapshot is written to MODEL_SUBDIR alongside the checkpoint.
"""

import argparse
import logging
import time
import warnings
from pathlib import Path

# Suppress known false-positive / deprecation warnings
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask")
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

import yaml
import wandb
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    MODEL_DIR, MODEL_NAME, MODEL_SUBDIR, N_TEMPORAL_BINS, HISTORY_WINDOW,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH,
    TRAIN_LR, TRAIN_D_MODEL, TRAIN_N_LAYERS, TRAIN_N_HEADS,
    TRAIN_MAX_SESSIONS, TRAIN_NUM_WORKERS,
    RQVAE_CODEBOOK_SIZE, RQVAE_N_LEVELS, SKU2CODES_PATH, RQVAE_MODEL_PATH,
    ITEM2VEC_PATH,
    ITEM_LOSS_ALPHA, ITEM_LABEL_SMOOTHING,
    CLEAN_PARQUET, TRAIN_CUTOFF,
)
from ingestion.dataset import InteractionGenerator
from simulation.generator.session_transformer import (
    SessionTransformer,
    N_ACTIONS,
    ITEM_BEARING_IDX,
    N_CODE_LEVELS,
)


def build_target_mask(lengths: torch.Tensor, T_out: int, device) -> torch.Tensor:
    """
    True at positions that are REAL targets (not padding).
    For session of length L, real output positions are 0..L-2
    (predicting tokens 1..L-1 from inputs 0..L-2).
    """
    positions = torch.arange(T_out, device=device).unsqueeze(0)   # [1, T_out]
    return positions < (lengths.to(device).unsqueeze(1) - 1)       # [B, T_out]


def build_item_loss_weights(device, alpha: float) -> list:
    """
    Per-level inverse-frequency weights for item CE loss (popularity debiasing).

    For each RQ-VAE code level k, count how often each code c appears as a target
    in training data. Weight w_k[c] = 1 / (freq + 1)^alpha, normalized so that
    mean weight = 1 (keeps the loss magnitude comparable across alpha values).

    freq[code_k] = sum of SKU frequencies over all SKUs whose triple has
    sku2codes[sku][k] == code. This matches the gradient distribution seen
    during training (events are sampled from event frequency, not SKU uniqueness).

    alpha=0   → uniform (disabled)
    alpha=0.5 → sqrt-inverse frequency (standard recsys long-tail fix)
    alpha=1.0 → full inverse frequency (aggressive)

    Returns:
        list of N_CODE_LEVELS tensors, each shape [RQVAE_CODEBOOK_SIZE] on device
    """
    import polars as pl
    from ingestion.rqvae import load_sku2codes

    if alpha <= 0:
        print(f"  item_loss_alpha={alpha} → uniform weights (debiasing disabled)")
        return [torch.ones(RQVAE_CODEBOOK_SIZE, device=device) for _ in range(N_CODE_LEVELS)]

    sku2codes, _ = load_sku2codes()

    freq_df = (
        pl.scan_parquet(str(CLEAN_PARQUET))
        .filter(pl.col("timestamp") < pl.lit(TRAIN_CUTOFF).str.to_datetime())
        .filter(pl.col("sku").is_not_null())
        .group_by("sku")
        .agg(pl.len().alias("freq"))
        .collect()
    )
    sku_freq = {int(r["sku"]): int(r["freq"]) for r in freq_df.iter_rows(named=True)}

    code_freq = torch.zeros(N_CODE_LEVELS, RQVAE_CODEBOOK_SIZE, dtype=torch.float64)
    for sku, triple in sku2codes.items():
        f = sku_freq.get(sku, 0)
        if f == 0:
            continue
        for k in range(N_CODE_LEVELS):
            code_freq[k, triple[k]] += f

    weights = []
    for k in range(N_CODE_LEVELS):
        w = 1.0 / (code_freq[k] + 1.0) ** alpha
        w = w / w.mean()                                          # normalize so mean=1
        weights.append(w.to(device=device, dtype=torch.float32))
        nz  = int((code_freq[k] > 0).sum().item())
        mn, mx = float(w.min()), float(w.max())
        print(f"  item_loss_weights[level {k}]: {nz}/{RQVAE_CODEBOOK_SIZE} codes active, "
              f"weight range [{mn:.3f}, {mx:.3f}]")

    return weights


def _ensure_prerequisites():
    """Train RQ-VAE and CTGAN if their artifacts are not already present."""
    from config import MODEL_DIR

    # RQ-VAE — required for dataset item encoding
    if SKU2CODES_PATH.exists():
        print(f"  sku2codes found — skipping RQ-VAE training.")
    else:
        print("\n  sku2codes not found — training RQ-VAE first ...")
        from ingestion.rqvae import train_rqvae, build_sku2codes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = train_rqvae(device=device)
        build_sku2codes(model, device=device)

    # CTGAN — required for session generation seeding at evaluation time
    ctgan_path = MODEL_DIR / "identity_sampler.pkl"
    if ctgan_path.exists():
        print(f"  identity_sampler found — skipping CTGAN training.")
    else:
        print("\n  identity_sampler not found — training CTGAN ...")
        from simulation.identity.CTGAN import IdentityFactory
        factory = IdentityFactory()
        factory.fit()


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    _ensure_prerequisites()
    print(f"Config: epochs={TRAIN_EPOCHS}, batch={TRAIN_BATCH_SIZE}, "
          f"d_model={TRAIN_D_MODEL}, layers={TRAIN_N_LAYERS}, heads={TRAIN_N_HEADS}, "
          f"lr={TRAIN_LR}, max_sessions={TRAIN_MAX_SESSIONS}")

    run = wandb.init(
        project="thesis-session-transformer",
        name=MODEL_NAME,
        config={
            "model":              MODEL_NAME,
            "epochs":             TRAIN_EPOCHS,
            "batch_size":         TRAIN_BATCH_SIZE,
            "max_length":         TRAIN_MAX_LENGTH,
            "lr":                 TRAIN_LR,
            "d_model":            TRAIN_D_MODEL,
            "n_layers":           TRAIN_N_LAYERS,
            "n_heads":            TRAIN_N_HEADS,
            "rqvae_codebook_size":  RQVAE_CODEBOOK_SIZE,
            "rqvae_n_levels":       RQVAE_N_LEVELS,
            "n_temporal_bins":      N_TEMPORAL_BINS,
            "history_window":       HISTORY_WINDOW,
            "max_sessions":         TRAIN_MAX_SESSIONS,
            "item_loss_alpha":      ITEM_LOSS_ALPHA,
            "item_label_smoothing": ITEM_LABEL_SMOOTHING,
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

    # Model
    model = SessionTransformer(
        codebook_size=RQVAE_CODEBOOK_SIZE,
        n_code_levels=RQVAE_N_LEVELS,
        d_model=TRAIN_D_MODEL,
        n_layers=TRAIN_N_LAYERS,
        n_heads=TRAIN_N_HEADS,
        n_temporal_bins=N_TEMPORAL_BINS,
        window_H=HISTORY_WINDOW,
        valid_transitions={},     # mask not used during training
    ).to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    print(f"\nBuilding per-level inverse-freq weights for item CE "
          f"(alpha={ITEM_LOSS_ALPHA}, label_smoothing={ITEM_LABEL_SMOOTHING}) ...")
    item_loss_weights = build_item_loss_weights(device, alpha=ITEM_LOSS_ALPHA)

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
            codes    = batch["codes"].to(device)                # [B, T, N_CODE_LEVELS]
            deltas   = batch["deltas"].to(device)               # [B, T]
            pad_mask = batch["tgt_key_padding_mask"].to(device) # [B, T-1]
            lengths  = batch["lengths"]                         # [B]
            history  = batch.get("history")                     # dict or None
            if history is not None:
                history = {k: v.to(device) for k, v in history.items()}

            B, T = events.shape
            T_out = T - 1

            # Targets
            tgt_actions = events[:, 1:]         # [B, T-1]
            tgt_codes   = codes[:, 1:]          # [B, T-1, N_CODE_LEVELS]
            tgt_deltas  = deltas[:, 1:]         # [B, T-1]

            # Valid position mask (real, non-padded targets)
            valid = build_target_mask(lengths, T_out, device)  # [B, T-1]

            # Item mask: valid positions where target is an item-bearing event
            tgt_is_item = torch.zeros_like(tgt_actions, dtype=torch.bool)
            for idx in ITEM_BEARING_IDX:
                tgt_is_item |= (tgt_actions == idx)
            item_valid = valid & tgt_is_item  # [B, T-1]

            # Forward (teacher-forced); item_logits_list[k] is sparse [M, codebook_size]
            with autocast("cuda"):
                action_logits, item_logits_list, temporal_logits = model(
                    events, codes, deltas,
                    history=history,
                    tgt_key_padding_mask=pad_mask,
                    item_mask=item_valid,
                )
                # action_logits:    [B, T-1, n_actions]
                # item_logits_list: list of N_CODE_LEVELS tensors [M, codebook_size]
                # temporal_logits:  [B, T-1, n_bins]

                # Action loss
                a_logits    = action_logits[valid]          # [N, N_ACTIONS]
                a_tgt       = tgt_actions[valid]            # [N]
                action_loss = F.cross_entropy(a_logits, a_tgt)

                # Item loss: mean CE over N_CODE_LEVELS code levels
                # Inverse-freq weights debiases popular codes; label smoothing prevents
                # over-confident concentration on a handful of triples.
                if item_valid.any():
                    item_loss = sum(
                        F.cross_entropy(
                            item_logits_list[k],
                            tgt_codes[:, :, k][item_valid],
                            weight=item_loss_weights[k],
                            label_smoothing=ITEM_LABEL_SMOOTHING,
                        )
                        for k in range(N_CODE_LEVELS)
                    ) / N_CODE_LEVELS
                else:
                    item_loss = torch.zeros(1, device=device).squeeze()

                # Temporal loss
                d_logits      = temporal_logits[valid]      # [N, N_BINS]
                d_tgt         = tgt_deltas[valid]           # [N]
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

        elapsed      = time.time() - t0
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
                codes    = batch["codes"].to(device)
                deltas   = batch["deltas"].to(device)
                pad_mask = batch["tgt_key_padding_mask"].to(device)
                lengths  = batch["lengths"]
                history  = batch.get("history")
                if history is not None:
                    history = {k: v.to(device) for k, v in history.items()}

                B, T  = events.shape
                T_out = T - 1
                tgt_actions = events[:, 1:]
                tgt_codes   = codes[:, 1:]
                tgt_deltas  = deltas[:, 1:]
                valid       = build_target_mask(lengths, T_out, device)

                tgt_is_item = torch.zeros_like(tgt_actions, dtype=torch.bool)
                for idx in ITEM_BEARING_IDX:
                    tgt_is_item |= (tgt_actions == idx)
                item_valid = valid & tgt_is_item

                with autocast("cuda"):
                    action_logits, item_logits_list, temporal_logits = model(
                        events, codes, deltas,
                        history=history,
                        tgt_key_padding_mask=pad_mask,
                        item_mask=item_valid,
                    )
                    a_logits = action_logits[valid]
                    val_action_loss = F.cross_entropy(a_logits, tgt_actions[valid])

                    if item_valid.any():
                        val_item_loss = sum(
                            F.cross_entropy(
                                item_logits_list[k],
                                tgt_codes[:, :, k][item_valid],
                                weight=item_loss_weights[k],
                                label_smoothing=ITEM_LABEL_SMOOTHING,
                            )
                            for k in range(N_CODE_LEVELS)
                        ) / N_CODE_LEVELS
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete RQ-VAE and CTGAN artifacts and rebuild them from scratch.")
    args = parser.parse_args()

    if args.rebuild:
        for path in [ITEM2VEC_PATH, SKU2CODES_PATH, RQVAE_MODEL_PATH, MODEL_DIR / "identity_sampler.pkl"]:
            if path.exists():
                path.unlink()
                print(f"  Deleted {path}")
            else:
                print(f"  Already absent: {path}")

    train()
