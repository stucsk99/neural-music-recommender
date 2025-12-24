from functools import lru_cache
import numpy as np
import tensorflow as tf
from app.configs import MODEL_PATH


@lru_cache(maxsize=1)
def get_infer():
    loaded = tf.saved_model.load(str(MODEL_PATH))
    return loaded.signatures["serving_default"]


def embed_mel_chunks(mel_chunks: np.ndarray) -> np.ndarray:
    """
    mel_chunks: shape (n_chunks, n_mels, chunk_length)
    Returns a single L2-normalized embedding vector (embedding_dim,)
    """
    if mel_chunks.ndim != 3:
        raise ValueError(f"Expected (n_chunks, n_mels, chunk_length), got {mel_chunks.shape}")

    X = mel_chunks[..., np.newaxis].astype("float32")  # (n_chunks, 128, 129, 1)

    infer = get_infer()

    # Call signature using the required input name: 'input_layer'
    outputs = infer(input_layer=tf.constant(X))

    # Read the required output key
    E = outputs["output_0"].numpy()  # (n_chunks, embedding_dim)

    e = E.mean(axis=0)
    e /= (np.linalg.norm(e) + 1e-8)
    return e
