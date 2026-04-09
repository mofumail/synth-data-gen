"""
main.py — train then evaluate in one command.

Usage:
    PYTHONPATH=. uv run python main.py
    PYTHONPATH=. uv run python main.py --rebuild
    PYTHONPATH=. uv run python main.py --skip-train --fidelity-only --n-sessions 1000
    PYTHONPATH=. uv run python main.py --final

Flags:
    --rebuild       Delete RQ-VAE + CTGAN artifacts before training.
    --skip-train    Skip training; run evaluation on the existing checkpoint.

All other flags are forwarded to evaluate.py (see evaluate.py for details).
"""

import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Train SessionTransformer then evaluate.")

    # Train flags
    p.add_argument("--rebuild",     action="store_true",
                   help="Delete RQ-VAE and CTGAN artifacts and rebuild from scratch.")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training; jump straight to evaluation.")

    # Eval flags (mirrors evaluate.py)
    p.add_argument("--max-train",     type=int, default=None)
    p.add_argument("--max-val",       type=int, default=None)
    p.add_argument("--max-test",      type=int, default=None)
    p.add_argument("--num-seeds",     type=int, default=5)
    p.add_argument("--base-seed",     type=int, default=42)
    p.add_argument("--n-sessions",    type=int, default=1000)
    p.add_argument("--k",             type=int, default=10)
    p.add_argument("--final",         action="store_true",
                   help="Use locked test split (run ONCE, after all model selection).")
    p.add_argument("--fidelity-only", action="store_true",
                   help="Skip GRU4Rec utility eval; fidelity metrics only.")

    return p.parse_args()


def main():
    args = parse_args()

    # --- Train ---
    if not args.skip_train:
        import warnings
        import logging
        warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask")
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

        import torch
        from config import SKU2CODES_PATH, RQVAE_MODEL_PATH, MODEL_DIR, ITEM2VEC_PATH

        if args.rebuild:
            for path in [ITEM2VEC_PATH, SKU2CODES_PATH, RQVAE_MODEL_PATH, MODEL_DIR / "identity_sampler.pkl"]:
                if path.exists():
                    path.unlink()
                    print(f"  Deleted {path}")
                else:
                    print(f"  Already absent: {path}")

        from train import train
        train()

    # --- Evaluate ---
    from evaluate import evaluate, run_fidelity_only
    if args.fidelity_only:
        run_fidelity_only(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
