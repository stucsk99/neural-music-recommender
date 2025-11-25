import io
import numpy as np
import librosa
from app.configs import SR, N_MELS, CHUNK_LENGTH
from app.embedding_utils import embed_mel_chunks

def mp3_bytes_to_mel_chunks(mp3_bytes: bytes):
    """
    Convert MP3 bytes to mel spectrogram chunks of shape (n_chunks, n_mels, chunk_length)
    Returns None if audio is too short.
    """
    y, sr = librosa.load(io.BytesIO(mp3_bytes), sr=SR)
    if y.size == 0:
        return None

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, n_frames)

    n_frames = mel_db.shape[1]
    if n_frames < CHUNK_LENGTH:
        return None

    chunks = [
        mel_db[:, s:s+CHUNK_LENGTH]
        for s in range(0, n_frames - CHUNK_LENGTH + 1, CHUNK_LENGTH)
    ]
    return np.stack(chunks)  # (n_chunks, n_mels, chunk_length)

def local_file_to_embedding(uploaded_file) -> np.ndarray:
    """
    Streamlit uploaded file → embedding vector.
    """
    mp3_bytes = uploaded_file.read()
    mel_chunks = mp3_bytes_to_mel_chunks(mp3_bytes)
    if mel_chunks is None:
        raise ValueError("Audio too short or could not compute mel spectrogram.")
    emb = embed_mel_chunks(mel_chunks)
    return emb
