from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parents[1]

# Model path (adjust name if yours is different)
MODEL_PATH = BASE_DIR / "models" / "embedding_model_contrastiveone_ep.keras"

# Data paths (Spotify index)
DATA_DIR = BASE_DIR / "data"
SPOTIFY_EMBS_PATH = DATA_DIR / "spotify_embs.npy"
SPOTIFY_FEATS_PATH = DATA_DIR / "spotify_feats_scaled.npy"
SPOTIFY_META_PATH = DATA_DIR / "spotify_meta.npy"
SCALER_PATH = DATA_DIR / "scaler.pkl"

FMA_EMBS_PATH = DATA_DIR / "fma_embs.npy"
FMA_META_PATH = DATA_DIR / "fma_meta.npy"

# Audio / spectrogram params (match your training setup)
SR = 22050
N_MELS = 128
CHUNK_LENGTH = 129  # frames

# Recommendation weights
ALPHA = 0.8  # weight for your neural embedding
BETA = 0.2   # weight for Spotify audio features (for future use when query has features)
