# Session-generator pipeline

Transformer-based generator of synthetic e-commerce interaction sessions (Synerise RecSys 2025 data), evaluated on fidelity, validity, and popularity bias against a Markov baseline and a real-vs-real (TRTR) reference. This README covers how to run the pipeline and where each stage's logic lives; design rationale and results are in the thesis (`thesis.tex`) and the wiki pages (Onderzoek, Ontwerp, Ontwikkeling, Evaluatie, Feedback & Iteratie).

## Setup

- Requires [uv](https://docs.astral.sh/uv/). Run `uv sync` once after cloning.
- Raw Synerise parquets are expected at `../DATA/` relative to `code/pipeline/` (edit `DATA_DIR` in `config.py` to change).
- All knobs live in `config.yaml` (dataset bounds, splits, sessionization, vocabulary, model size, item-head iteration, SVD-PQ, sampling temperatures, bias-mitigation levers, evaluation scale). The full surface is documented in the wiki page *Ontwikkeling* §Hyperparameter Surface.

## Running

End-to-end (recommended):

```sh
uv run python main.py                      # all four stages: preprocess -> svdpq -> train -> evaluate
uv run python main.py --from train         # this stage and everything after it
uv run python main.py --stages preprocess,svdpq
uv run python main.py --comment my_run     # tag appended to the output folder name
```

Evaluation flags (forwarded by `main.py`, or pass directly to `evaluate.py`): `--n-sessions <int>` (synthetic sessions per seed), `--num-seeds <int>`. Final-evaluation defaults come from `config.yaml` (`final_eval_n_sessions: 500000`, `final_eval_seeds: [42..46]`).

Stage-by-stage equivalents:

```sh
uv run python preprocess.py            # cleaning, sessionization, time-aware split, vocab, constraint extraction
uv run python ingestion/svdpq.py       # offline SVD-PQ item tokenizer (only needed for item_head_mode: svdpq)
uv run python train.py                 # teacher-forced multi-head training
uv run python evaluate.py              # generation + fidelity/validity evaluation vs Markov + TRTR
```

## Four-stage architecture

Stages communicate via durable on-disk artefacts, so any stage can be re-run without re-running its predecessors (the artefact interfaces are drawn in the thesis C4 diagrams):

1. **Ingestion** (`preprocess.py`, `ingestion/`) — schema normalization, deduplication, 30-min sessionization, time-aware split (train cutoff 2022-11-15, val cutoff 2022-12-01), top-K vocabulary (K=200k), valid-transition and exposure-constraint extraction, SVD-PQ codebook construction (`ingestion/svdpq.py`). Outputs cleaned parquets + vocab + codebooks under `output/`.
2. **Training** (`train.py`, `models/`) — multi-head autoregressive transformer (action / category / item / temporal heads over a shared causal decoder, d=128, 4 layers, 4 heads); item head per iteration: `flat`, `hier`, or `svdpq` (`item_head_mode` in `config.yaml`). Checkpoint selection by validation loss (the in-training TSTR probe exists but is disabled: `is_best_eval_enabled: false`). Outputs `output/models/session_transformer_<hyperparams>_<timestamp>/`.
3. **Generation runtime** (inside `evaluate.py` / `generate_baskets.py`) — autoregressive sampling driven by (user_id, session_id) arrivals, with the hard-constraint validity layer (illegal transitions, timestamp monotonicity, purchase-exposure repair).
4. **Evaluation** (`evaluation/`) — fidelity suite (`fidelity.py`), validity checks (`validity_checker.py`), Markov baseline (`markov.py`), TRTR reference (`reference.py`), HTML report (`report.py`), orchestration (`orchestrator.py`). The archived TSTR evaluator is `downstream.py` + `in_train_tstr.py`.

## Popularity-bias evaluation (basket protocol)

Group-level popularity-bias metrics (within-group Gini, ΔGAP, between-group GAP, group cosine; Braun, Bhaumik & Dey 2023) over per-user recommendation baskets:

```sh
uv run python generate_baskets.py            # per-iteration anchored per-user baskets (10k users/seed)
uv run python generate_reference_baskets.py  # real-continuation reference baskets for the same users
uv run python evaluate_popularity_bias.py    # metrics; defaults: --group-by profile_pop --n-groups 2 (median split)
uv run python plot_popularity_bias.py        # figures (popbias_dgap, popbias_diversity)
```

Implementation: `evaluation/popularity_bias.py` (metrics), `evaluation/baskets.py` (basket assembly). Results land in `output/popularity_bias/` as CSV + JSON per lens (`rollout-*` = sampled sessions, `topk-*` = top-k head, `real-*` = reference). Metric definitions, adaptations to this catalog, and interpretation are in the wiki page *Evaluatie*.

## Output layout

- `output/` — cleaned parquets, vocabularies, category pools, codebooks, catalog metadata.
- `output/models/<run>/` — checkpoints, generation logs, evaluation HTML reports, `baskets/`.
- `output/popularity_bias/` — basket-level bias metrics per run/lens.

Output folder names encode hyperparameters and the run datetime; `--comment` appends a free-text tag.
