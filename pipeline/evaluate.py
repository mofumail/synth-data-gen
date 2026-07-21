"""
evaluate.py - End-to-end evaluation CLI

Loads real data, builds reference store, runs EvaluationOrchestrator across
seeds, saves HTML + JSON report.

Split discipline:
    Default (development / ablations):
        Downstream GRU4Rec is tested on val_split (Nov 15 – Dec 1).
        Use this for all model comparisons and ablation runs.

    --final (final thesis numbers only, run ONCE):
        Downstream GRU4Rec is tested on test_split (Dec 1 – Dec 8).
        Do NOT use this until all model selection is complete.

Usage (full development run):
    PYTHONPATH=. uv run python evaluate.py

Usage (final locked-test run):
    PYTHONPATH=. uv run python evaluate.py --final

Usage (fidelity-only, fast iteration):
    PYTHONPATH=. uv run python evaluate.py --fidelity-only \\
        --max-train 50000 --max-val 10000 --n-sessions 1000

Usage (smoke test):
    PYTHONPATH=. uv run python evaluate.py \\
        --max-train 10000 --max-val 2000 --n-sessions 500 --num-seeds 1

Reports saved to OUTPUT_DIR/reports/ (HTML + JSON).
"""

import argparse
import time
import torch
from collections import defaultdict

from config import MODEL_DIR, EVAL_MODEL_SUBDIR, EVAL_MODEL_NAME, OUTPUT_DIR
from evaluation.fidelity import FidelityEvaluator
from evaluation.orchestrator import EvaluationOrchestrator
from evaluation.reference import RealDataLoader, ReferenceProfiler, ReferenceStore
from evaluation.report import ReportGenerator
from evaluation.markov import MarkovSessionGenerator
from simulation.generator.session_generator import SessionGenerator
from simulation.validity import ValidityLayer


REPORTS_DIR        = EVAL_MODEL_SUBDIR / "reports"
REF_STORE_PATH     = OUTPUT_DIR / "reference_store.joblib"
VAL_REF_STORE_PATH = OUTPUT_DIR / "val_reference_store.joblib"


def build_ref_store(real_data) -> tuple:
    """Load reference stores from cache if available, otherwise build and save. Returns (train_rs, val_rs)."""
    profiler = ReferenceProfiler()

    if REF_STORE_PATH.exists():
        print(f"\nLoading train ReferenceStore from cache ...")
        train_rs = ReferenceStore.load(REF_STORE_PATH)
    else:
        print("\nBuilding train ReferenceStore ...")
        train_rs = profiler.profile(real_data.train_split)
        train_rs.save(REF_STORE_PATH)

    if VAL_REF_STORE_PATH.exists():
        print(f"\nLoading val ReferenceStore from cache ...")
        val_rs = ReferenceStore.load(VAL_REF_STORE_PATH)
    else:
        print("\nBuilding val ReferenceStore (generalization check) ...")
        val_rs = profiler.profile(real_data.val_split)
        val_rs.save(VAL_REF_STORE_PATH)

    return train_rs, val_rs


def ensure_identity_sampler() -> None:
    """Train IdentityFactory (CTGAN) if no sampler exists yet."""
    from simulation.identity.CTGAN import IdentityFactory
    sampler_path = MODEL_DIR / "identity_sampler.pkl"
    if sampler_path.exists():
        print("  Identity sampler found, skipping training.")
        return
    print("  Identity sampler not found — training CTGAN IdentityFactory...")
    factory = IdentityFactory(model_storage=sampler_path)
    factory.fit()


def build_transformer_generator(ref_store: ReferenceStore) -> SessionGenerator:
    model_path = EVAL_MODEL_SUBDIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run train.py first."
        )
    valid_transitions = defaultdict(set)
    for a, b in ref_store.legal_bigrams:
        valid_transitions[a].add(b)

    sampler_path   = MODEL_DIR / "identity_sampler.pkl"
    validity_layer = ValidityLayer(legal_bigrams=ref_store.legal_bigrams_set())
    device         = "cuda" if torch.cuda.is_available() else "cpu"
    return SessionGenerator(
        model_path            = model_path,
        validity_layer        = validity_layer,
        valid_transitions     = valid_transitions,
        identity_sampler_path = str(sampler_path) if sampler_path.exists() else None,
        device                = device,
    )


def evaluate(args):
    t0 = time.time()

    # --- Identity sampler ---
    print("\nChecking identity sampler ...")
    ensure_identity_sampler()

    # --- Real data ---
    print("\nLoading real data ...")
    real_data = RealDataLoader.load(
        max_train_sessions = args.max_train,
        max_val_sessions   = args.max_val,
        max_test_sessions  = args.max_test,
        include_test       = args.final,
    )

    # --- Reference stores ---
    ref_store, val_ref_store = build_ref_store(real_data)

    # --- Generators ---
    print("\nBuilding generators ...")
    primary_gen  = build_transformer_generator(ref_store)
    baseline_gen = MarkovSessionGenerator.from_parquet(max_length=50)
    print("  Generators ready.")

    # --- Orchestrate ---
    gru4rec_dir = MODEL_DIR / "gru4rec"
    orchestrator = EvaluationOrchestrator(
        num_seeds        = args.num_seeds,
        base_seed        = args.base_seed,
        k                = args.k,
        gru4rec_save_dir = str(gru4rec_dir),
        model_tag        = EVAL_MODEL_NAME,
    )
    aggregated = orchestrator.evaluate(
        primary_gen     = primary_gen,
        baseline_gen    = baseline_gen,
        real_data       = real_data,
        ref_store       = ref_store,
        val_ref_store   = val_ref_store,
        n_sessions      = args.n_sessions,
        use_test_split  = args.final,
    )

    # --- Report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reporter = ReportGenerator()
    html     = reporter.generate(aggregated)
    reporter.save(html,       REPORTS_DIR / "evaluation_report.html")
    reporter.serialize(aggregated, REPORTS_DIR / "evaluation_results.json")

    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print(f"Reports -> {REPORTS_DIR}")

    # Quick summary to stdout
    print("\n=== Summary ===")
    ft = aggregated.fidelity_train_mean
    fv = aggregated.fidelity_val_mean
    print(f"  Fidelity vs train | JSD(action)={ft.jsd_action:.4f}  KS(len)={ft.ks_session_length:.4f}  diversity={ft.sample_diversity:.4f}  L1(bigrams)={ft.l1_action_bigrams:.4f}  JSD(pop)={ft.bias.popularity_jsd:.4f}  coverage={ft.bias.item_coverage:.4f}  gini_delta={ft.bias.gini_coefficient_delta:.4f}")
    print(f"  Fidelity vs val   | JSD(action)={fv.jsd_action:.4f}  KS(len)={fv.ks_session_length:.4f}  diversity={fv.sample_diversity:.4f}  L1(bigrams)={fv.l1_action_bigrams:.4f}  JSD(pop)={fv.bias.popularity_jsd:.4f}  coverage={fv.bias.item_coverage:.4f}  gini_delta={fv.bias.gini_coefficient_delta:.4f}")
    for um in aggregated.utility_mean:
        print(f"  Utility   | {um.condition}: HR@{args.k}={um.hr_at_k:.4f}  NDCG@{args.k}={um.ndcg_at_k:.4f}")
    vpm = aggregated.validity_post_mean
    print(f"  Validity  | illegal={vpm.illegal_transition_rate:.4f}  monoton={vpm.monotonicity_violation_rate:.4f}  purchase={vpm.purchase_exposure_rate:.4f}")
    if aggregated.correlations:
        print("  Correlations (fidelity -> TSTR-T HR@K):")
        for c in aggregated.correlations:
            sig = "(*)" if c.p_value < 0.05 else ""
            print(f"    {c.fidelity_metric:<20} r={c.correlation:+.3f}  p={c.p_value:.3f} {sig}")


def run_fidelity_only(args):
    """Fast fidelity-only path: generate n_sessions, compute fidelity vs train + val, print."""
    t0 = time.time()

    print("\nChecking identity sampler ...")
    ensure_identity_sampler()

    print("\nLoading real data ...")
    real_data = RealDataLoader.load(
        max_train_sessions=args.max_train,
        max_val_sessions=args.max_val,
    )

    ref_store, val_ref_store = build_ref_store(real_data)

    print("\nBuilding generator ...")
    primary_gen = build_transformer_generator(ref_store)

    print(f"\nGenerating {args.n_sessions} sessions (seed={args.base_seed}) ...")
    synth = primary_gen.generate(
        n_sessions=args.n_sessions, seed=args.base_seed, apply_constraints=True
    )
    print(f"  {len(synth)} sessions generated")

    fid_ev = FidelityEvaluator()
    ft = fid_ev.evaluate(real_data, synth, ref_store)
    fv = fid_ev.evaluate(real_data, synth, val_ref_store)

    def _print(label, f):
        print(f"\n=== Fidelity {label} ===")
        print(f"  JSD(action)        = {f.jsd_action:.4f}")
        print(f"  KS(session_len)    = {f.ks_session_length:.4f}")
        print(f"  KS(temporal_delta) = {f.ks_temporal_delta:.4f}")
        print(f"  L1(action bigrams) = {f.l1_action_bigrams:.4f}")
        print(f"  L1(item bigrams)   = {f.l1_item_bigrams:.4f}")
        print(f"  Diversity (ratio)  = {f.sample_diversity:.4f}")
        print(f"  Conv. rate delta   = {f.conversion_rate_delta:.4f}")
        print(f"  Cart abandon delta = {f.cart_abandonment_delta:.4f}")
        print(f"  JSD(popularity)    = {f.bias.popularity_jsd:.4f}")
        print(f"  Item coverage      = {f.bias.item_coverage:.4f}")
        print(f"  Gini delta         = {f.bias.gini_coefficient_delta:.4f}")

    _print("vs Train", ft)
    _print("vs Val  ", fv)
    print(f"\nDone in {time.time() - t0:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate synthetic session generator")
    p.add_argument("--max-train",      type=int,   default=None,
                   help="Cap training sessions (default: all)")
    p.add_argument("--max-val",        type=int,   default=None,
                   help="Cap val sessions (default: all)")
    p.add_argument("--max-test",       type=int,   default=None,
                   help="Cap test sessions (default: all)")
    p.add_argument("--num-seeds",      type=int,   default=5)
    p.add_argument("--base-seed",      type=int,   default=42)
    p.add_argument("--n-sessions",     type=int,   default=1000,
                   help="Synthetic sessions to generate per seed")
    p.add_argument("--k",              type=int,   default=10,
                   help="Cutoff for HR@K and NDCG@K")
    p.add_argument("--final",          action="store_true",
                   help="Use locked test split (Dec 1–8) instead of val split. "
                        "Run ONCE only, after all model selection is done.")
    p.add_argument("--fidelity-only",  action="store_true",
                   help="Generate n_sessions synthetic sessions and compute fidelity metrics only. "
                        "Skips GRU4Rec utility evaluation. Single seed, fast.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fidelity_only:
        run_fidelity_only(args)
    else:
        evaluate(args)
