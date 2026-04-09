from pathlib import Path
import yaml
# Side note; these first 2 paths are specific for my own folder locations, you'd want to change these before 
# running anything
#Paths
DATA_DIR      = Path(__file__).parent.parent / "DATA"
PIPELINE_DIR  = Path(__file__).parent
OUTPUT_DIR    = PIPELINE_DIR / "output"
CLEAN_PARQUET = OUTPUT_DIR / "events_clean.parquet"
TEST_PARQUET  = OUTPUT_DIR / "events_test.parquet"
MODEL_DIR     = OUTPUT_DIR / "models"
SYNTH_DIR     = OUTPUT_DIR / "synthetic"

# Load config.yaml
_cfg = yaml.safe_load((PIPELINE_DIR / "config.yaml").read_text())

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

#Vocabulary — derived from sku2codes catalog after rqvae.py runs; falls back to config.yaml
_catalog_size_path = OUTPUT_DIR / "catalog_size.txt"
VOCAB_K = (
    int(_catalog_size_path.read_text().strip())
    if _catalog_size_path.exists()
    else _cfg["vocab_k"]
)

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
TRAIN_NUM_WORKERS  = _cfg.get("train_num_workers", 4)

# Derived: unique model name used for checkpoint and config snapshot filenames.
# Changing d_model or n_layers in config.yaml automatically routes to a different
# file so ablation variants never overwrite each other.
# RQ-VAE item tokeniser
SKU2CODES_PATH   = OUTPUT_DIR / "sku2codes.joblib"
RQVAE_MODEL_PATH = MODEL_DIR  / "item_rqvae.pt"

RQVAE_CODEBOOK_SIZE = _cfg["rqvae_codebook_size"]
RQVAE_N_LEVELS      = _cfg["rqvae_n_levels"]
RQVAE_LATENT_DIM    = _cfg["rqvae_latent_dim"]
RQVAE_EPOCHS        = _cfg["rqvae_epochs"]
RQVAE_LR            = _cfg["rqvae_lr"]
RQVAE_BATCH_SIZE    = _cfg["rqvae_batch_size"]

# item2vec
ITEM2VEC_PATH      = OUTPUT_DIR / "item2vec_embeddings.joblib"
ITEM2VEC_DIM       = _cfg.get("item2vec_dim",       64)
ITEM2VEC_EPOCHS    = _cfg.get("item2vec_epochs",    10)
ITEM2VEC_WINDOW    = _cfg.get("item2vec_window",     5)
ITEM2VEC_MIN_COUNT = _cfg.get("item2vec_min_count",  2)

# Long-tail diversity knobs
INFER_TEMPERATURE    = _cfg.get("infer_temperature",    1.2)
ITEM_LOSS_ALPHA      = _cfg.get("item_loss_alpha",      0.5)
ITEM_LABEL_SMOOTHING = _cfg.get("item_label_smoothing", 0.1)

MODEL_NAME   = f"session_transformer_d{TRAIN_D_MODEL}_l{TRAIN_N_LAYERS}_h{TRAIN_N_HEADS}_rqvae"
MODEL_SUBDIR = MODEL_DIR / MODEL_NAME   # output/models/<name>/  -one folder per variant

#Evaluation model (independent of training config)
EVAL_MODEL_NAME  = _cfg.get("eval_model", MODEL_NAME)   # defaults to current train model
EVAL_MODEL_SUBDIR = MODEL_DIR / EVAL_MODEL_NAME

#GRU4Rec downstream evaluator
GRU4REC_VOCAB_K     = _cfg.get("gru4rec_vocab_k", 50000)
GRU4REC_EMBED_DIM   = _cfg["gru4rec_embed_dim"]
GRU4REC_HIDDEN_DIM  = _cfg["gru4rec_hidden_dim"]
GRU4REC_MAX_EPOCHS  = _cfg["gru4rec_max_epochs"]
GRU4REC_BATCH_SIZE  = _cfg["gru4rec_batch_size"]
GRU4REC_LR          = _cfg["gru4rec_lr"]
GRU4REC_PATIENCE    = _cfg["gru4rec_patience"]
GRU4REC_LR_PATIENCE = _cfg["gru4rec_lr_patience"]
GRU4REC_VAL_SPLIT   = _cfg["gru4rec_val_split"]
GRU4REC_LR_FACTOR   = _cfg["gru4rec_lr_factor"]
