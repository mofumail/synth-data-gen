from pathlib import Path
import yaml
# Side note; these first 2 paths are specific for my own folder locations, you'd want to change these before 
# running anything
#Paths
DATA_DIR      = Path(__file__).parent.parent / "DATA"
PIPELINE_DIR  = Path(__file__).parent
OUTPUT_DIR    = PIPELINE_DIR / "output"
CLEAN_PARQUET      = OUTPUT_DIR / "events_clean.parquet"
TEST_PARQUET       = OUTPUT_DIR / "events_test.parquet"
MODEL_DIR          = OUTPUT_DIR / "models"
SYNTH_DIR          = OUTPUT_DIR / "synthetic"
CAT2IDX_PATH       = OUTPUT_DIR / "cat2idx.joblib"

# Load config.yaml
_cfg = yaml.safe_load((PIPELINE_DIR / "config.yaml").read_text())

# Vocab-keyed artifact paths: changing vocab_k routes to a different file so
# stale 630k caches don't get reused under a 200k config.
def _vocab_suffixed(name: str, ext: str) -> Path:
    return OUTPUT_DIR / f"{name}_v{_cfg['vocab_k']}.{ext}"

CAT_SKU_POOLS_PATH = _vocab_suffixed("cat_sku_pools", "joblib")
SVDPQ_PATH         = _vocab_suffixed("sku_tokens", "joblib")

#Dataset bounds
DS_START = _cfg["ds_start"]
DS_END   = _cfg["ds_end"]

#Train / val / test splits
TRAIN_CUTOFF = _cfg["train_cutoff"]
VAL_CUTOFF   = _cfg["val_cutoff"]

#Sessionization
SESSION_TIMEOUT_MIN = _cfg["session_timeout_min"]
TEMPORAL_MAX_S      = SESSION_TIMEOUT_MIN * 60   # derived: max intra-session delta (seconds)
HISTORY_WINDOW      = _cfg["history_window"]

#Vocabulary
VOCAB_K = _cfg["vocab_k"]

#Temporal head
N_TEMPORAL_BINS = _cfg["n_temporal_bins"]
TEMPORAL_MIN_S  = _cfg["temporal_min_s"]

#Data loading
PAGE_VISIT_SAMPLE = _cfg["page_visit_sample"]

#Event types
ITEM_BEARING_EVENTS = {"add_to_cart", "remove_from_cart", "product_buy"}
ALL_EVENT_TYPES     = {
    "page_visit", "search_query",
    "add_to_cart", "remove_from_cart", "product_buy",
}

#MongoDB (locally run through docker)
MONGO_URI = _cfg["mongo_uri"]
MONGO_DB  = _cfg["mongo_db"]

#Training
TRAIN_EPOCHS       = _cfg["train_epochs"]
TRAIN_BATCH_SIZE   = _cfg["train_batch_size"]
TRAIN_MAX_LENGTH   = _cfg["train_max_length"]
TRAIN_LR           = _cfg["train_lr"]
TRAIN_D_MODEL      = _cfg["train_d_model"]
TRAIN_N_LAYERS     = _cfg["train_n_layers"]
TRAIN_N_HEADS      = _cfg["train_n_heads"]
TRAIN_MAX_SESSIONS = _cfg["train_max_sessions"]   # None = full dataset
TRAIN_NUM_WORKERS  = _cfg["train_num_workers"]

#Category head (hierarchical item loss)
CATEGORY_RARE_THRESHOLD = _cfg["category_rare_threshold"]

#SVD-PQ item tokenizer
SVDPQ_ENABLED          = _cfg["svdpq_enabled"]
SVDPQ_T                = _cfg["svdpq_t"]
SVDPQ_V                = _cfg["svdpq_v"]
SVDPQ_BINNING          = _cfg["svdpq_binning"]
SVDPQ_NOISE_STD        = _cfg["svdpq_noise_std"]
SVDPQ_MIN_INTERACTIONS = _cfg["svdpq_min_interactions"]
SVDPQ_EVENT_WEIGHTS    = _cfg["svdpq_event_weights"]
SVDPQ_LABEL_SMOOTHING  = float(_cfg.get("svdpq_label_smoothing", 0.0))
if not 0.0 <= SVDPQ_LABEL_SMOOTHING < 1.0:
    raise ValueError(
        f"svdpq_label_smoothing must be in [0, 1), got {SVDPQ_LABEL_SMOOTHING}"
    )

#Inference sampling
INFER_TEMPERATURE      = _cfg["infer_temperature"]
INFER_ITEM_TEMPERATURE = _cfg["infer_item_temperature"]
SVDPQ_INFER_SCORER     = _cfg.get("svdpq_infer_scorer", "hamming")
if SVDPQ_INFER_SCORER not in ("hamming", "log_prob"):
    raise ValueError(
        f"svdpq_infer_scorer must be 'hamming' or 'log_prob', got {SVDPQ_INFER_SCORER!r}"
    )

def _load_n_categories() -> int:
    """Load n_categories from cat2idx.joblib if available (built by preprocess.py)."""
    if CAT2IDX_PATH.exists():
        import joblib as _jl
        cat2idx = _jl.load(CAT2IDX_PATH)
        # indices: 0=PAD, 1=RARE, 2..N=regular; max_idx + 1 = n_categories
        return max(cat2idx.values()) + 1 if cat2idx else 2
    return 2   # fallback: PAD + RARE only (will fail at training if not rebuilt)

N_CATEGORIES = _load_n_categories()

# Derived unique model name used for checkpoint and config snapshot filenames.
# Changing d_model or n_layers in config.yaml automatically routes to a different
# file so ablation variants never overwrite each other.
_MODEL_VARIANT = f"svdpq_t{SVDPQ_T}v{SVDPQ_V}" if SVDPQ_ENABLED else "hier"
MODEL_NAME   = f"session_transformer_d{TRAIN_D_MODEL}_l{TRAIN_N_LAYERS}_h{TRAIN_N_HEADS}_{_MODEL_VARIANT}"
MODEL_SUBDIR = MODEL_DIR / MODEL_NAME   # output/models/<name>/

#Evaluation model: the specific timestamped training-run folder to evaluate,
#e.g. "session_transformer_d128_l4_h4_svdpq_t4v512_210426-15-58-18". Set via
#the `eval_model` key in config.yaml. Left as None at import time so the
#various scripts that import config.py (train, notebooks, ...) don't fail
#when the key is absent; evaluate.py / simulation/orchestrator.py raise a
#clear error the first time they actually need the path.
EVAL_MODEL_NAME   = _cfg.get("eval_model")
EVAL_MODEL_SUBDIR = MODEL_DIR / EVAL_MODEL_NAME if EVAL_MODEL_NAME else None

#GRU4Rec downstream evaluator
GRU4REC_VOCAB_K     = _cfg["gru4rec_vocab_k"]
GRU4REC_EMBED_DIM   = _cfg["gru4rec_embed_dim"]
GRU4REC_HIDDEN_DIM  = _cfg["gru4rec_hidden_dim"]
GRU4REC_MAX_EPOCHS  = _cfg["gru4rec_max_epochs"]
GRU4REC_BATCH_SIZE  = _cfg["gru4rec_batch_size"]
GRU4REC_LR          = _cfg["gru4rec_lr"]
GRU4REC_PATIENCE    = _cfg["gru4rec_patience"]
GRU4REC_LR_PATIENCE = _cfg["gru4rec_lr_patience"]
GRU4REC_VAL_SPLIT   = _cfg["gru4rec_val_split"]
GRU4REC_LR_FACTOR   = _cfg["gru4rec_lr_factor"]
