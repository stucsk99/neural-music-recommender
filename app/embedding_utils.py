from functools import lru_cache
import numpy as np
import tensorflow as tf
from app.configs import MODEL_PATH

# Import l2_normalize so Keras can deserialize it from the saved model
# (it's decorated with @register_keras_serializable() in src.model)
try:
    from src.model import l2_normalize
except ImportError:
    pass  # If src is not available, model loading will still work if l2_normalize is registered

@lru_cache(maxsize=1)
def get_embedding_model():
    # Load the model. The @register_keras_serializable() decorator on l2_normalize
    # in src/model.py makes safe_mode=False unnecessary, but we keep it for safety.
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    return model

def embed_mel_chunks(mel_chunks: np.ndarray) -> np.ndarray:
    """
    mel_chunks: shape (n_chunks, n_mels, chunk_length)
    Returns a single L2-normalized embedding vector (embedding_dim,)
    """
    if mel_chunks.ndim != 3:
        raise ValueError(f"Expected (n_chunks, n_mels, chunk_length), got {mel_chunks.shape}")

    model = get_embedding_model()
    X = mel_chunks[..., np.newaxis].astype("float32")  # (n_chunks, n_mels, chunk_length, 1)
    E = model.predict(X, verbose=0)                    # (n_chunks, embedding_dim)
    e = E.mean(axis=0)                                 # aggregate over chunks
    e /= (np.linalg.norm(e) + 1e-8)                    # L2-normalize
    return e
