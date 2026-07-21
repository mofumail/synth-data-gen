"""End-to-end pipeline orchestrator.

Runs preprocess -> svdpq -> train -> evaluate as subprocesses so each stage
gets a fresh config.py import (N_CATEGORIES and EVAL_MODEL_SUBDIR resolve
against the latest on-disk state).

Usage:
    python main.py                          # run all four stages
    python main.py --stages preprocess,svdpq
    python main.py --from train             # run train + evaluate
    python main.py --n-sessions 1000 --fidelity-only   # forwarded to evaluate.py
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).parent
CONFIG_YAML  = PIPELINE_DIR / "config.yaml"

STAGES = ["preprocess", "svdpq", "train", "evaluate"]

STAGE_CMDS = {
    "preprocess": ["preprocess.py"],
    "svdpq":      ["ingestion/svdpq.py"],
    "train":      ["train.py"],
    "evaluate":   ["evaluate.py"],
}


def run_stage(name: str, extra_args: list[str] | None = None) -> None:
    script = STAGE_CMDS[name]
    cmd = [sys.executable, *script, *(extra_args or [])]
    print(f"\n===== [{name}] {' '.join(cmd)} =====", flush=True)
    try:
        subprocess.run(cmd, cwd=PIPELINE_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[main] stage '{name}' failed with exit code {e.returncode}",
              file=sys.stderr)
        sys.exit(e.returncode)


def latest_trained_model() -> str:
    """Return the folder name of the newest {MODEL_NAME}_* run with a model.pt."""
    # Fresh import: preprocess may have just written cat2idx.joblib, changing
    # N_CATEGORIES; train may have just written a new run folder.
    import importlib
    import config as _config
    importlib.reload(_config)

    candidates = [
        p for p in _config.MODEL_DIR.glob(f"{_config.MODEL_NAME}_*")
        if p.is_dir() and (p / "model.pt").exists()
    ]
    if not candidates:
        raise RuntimeError(
            f"No trained model found under {_config.MODEL_DIR} matching "
            f"'{_config.MODEL_NAME}_*' with a model.pt. Did train.py succeed?"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def update_eval_model(folder_name: str) -> None:
    """Line-level edit so comments and formatting in config.yaml survive."""
    text = CONFIG_YAML.read_text()
    cfg = yaml.safe_load(text)
    if cfg.get("eval_model") == folder_name:
        return
    pattern = re.compile(r"^(eval_model\s*:).*$", re.MULTILINE)
    new_text, n = pattern.subn(f"\\1 {folder_name}", text, count=1)
    if n == 0:
        if not new_text.endswith("\n"):
            new_text += "\n"
        new_text += f"eval_model: {folder_name}\n"
    CONFIG_YAML.write_text(new_text)
    print(f"[main] set eval_model = {folder_name} in config.yaml")


def parse_args() -> tuple[list[str], list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma-separated subset of {STAGES}. Default: all.",
    )
    p.add_argument(
        "--from", dest="from_stage", choices=STAGES, default=None,
        help="Run this stage and every stage after it. Overrides --stages.",
    )
    # Everything unknown is forwarded to evaluate.py.
    ns, forwarded = p.parse_known_args()

    if ns.from_stage:
        idx = STAGES.index(ns.from_stage)
        stages = STAGES[idx:]
    else:
        stages = [s.strip() for s in ns.stages.split(",") if s.strip()]
        bad = [s for s in stages if s not in STAGES]
        if bad:
            p.error(f"unknown stage(s): {bad}. Valid: {STAGES}")
    return stages, forwarded


def main() -> None:
    stages, eval_args = parse_args()
    print(f"[main] plan: {' -> '.join(stages)}")
    if eval_args:
        print(f"[main] evaluate args: {eval_args}")

    for stage in stages:
        if stage == "evaluate":
            # If train ran in this invocation, or eval_model is unset, pick the
            # newest trained run and write it to config.yaml.
            cfg = yaml.safe_load(CONFIG_YAML.read_text())
            if "train" in stages or not cfg.get("eval_model"):
                update_eval_model(latest_trained_model())
            run_stage("evaluate", eval_args)
        else:
            run_stage(stage)

    print("\n[main] pipeline complete.")


if __name__ == "__main__":
    main()
