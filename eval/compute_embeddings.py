import os, numpy as np, tensorflow as tf
import traceback
import time

# Build model from code (safe) and load weights from the saved model
# directory. This avoids deserializing a Python lambda from the saved
# model and is the recommended safe workflow.
MODEL_PATH = "E:/song_project/models/embedding_model_contrastive.keras"
SPEC_DIR   = "E:/song_project/spectrograms"  # your .npz files

# Ensure src is on sys.path so we can import the builder
import sys
script_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tensorflow import keras
# Import l2_normalize so Keras can find it when deserializing the model
from model import l2_normalize

# Load the saved Keras model directly. The saved model contains a Lambda
# layer defined with a Python lambda, which Keras disallows by default
# for deserialization because it can execute arbitrary code. You said you
# trust the model source, so we pass `safe_mode=False` to allow loading.
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

def file_to_embedding(npz_path, chunk_length=129):
    data = np.load(npz_path)
    mel = data["mel"]  # (n_mels, n_frames) OR (n_chunks, n_mels, chunk_length)

    # Ensure shape is (n_chunks, n_mels, chunk_length)
    if mel.ndim == 2:
        n_frames = mel.shape[1]
        chunks = []
        for s in range(0, n_frames - chunk_length + 1, chunk_length):
            chunks.append(mel[:, s:s+chunk_length])
        if not chunks:  # too short
            return None
        mel = np.stack(chunks)

    X = mel[..., np.newaxis].astype("float32")  # (n_chunks, 128, 129, 1)
    emb = model.predict(X, verbose=0)           # (n_chunks, embed_dim)
    emb = emb.mean(axis=0)                      # (embed_dim,)
    emb = emb / (np.linalg.norm(emb) + 1e-8)    # L2-normalize
    return emb


print(f"[INFO] Starting embedding computation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[INFO] Model: {MODEL_PATH}")
print(f"[INFO] Spectrogram directory: {SPEC_DIR}")

all_files = [f for f in os.listdir(SPEC_DIR) if f.endswith('.npz')]
print(f"[INFO] Found {len(all_files)} .npz files to process.")

paths, embs, errors = [], [], []
for idx, f in enumerate(all_files):
    npz_path = os.path.join(SPEC_DIR, f)
    print(f"[INFO] ({idx+1}/{len(all_files)}) Processing: {f}")
    try:
        e = file_to_embedding(npz_path)
        if e is not None:
            paths.append(f)
            embs.append(e)
            print(f"[OK]   Embedded: {f} -> shape {e.shape}")
        else:
            print(f"[WARN] Skipped (too short or empty): {f}")
    except Exception as ex:
        print(f"[ERROR] Failed to process {f}: {ex}")
        traceback.print_exc()
        errors.append((f, str(ex)))

if embs:
    embs = np.stack(embs)  # (n_songs, embed_dim)
    np.save("E:/song_project/embeddings.npy", embs)
    np.save("E:/song_project/embeddings_files.npy", np.array(paths))
    print(f"[INFO] Saved embeddings to E:/song_project/embeddings.npy")
    print(f"[INFO] Saved file list to E:/song_project/embeddings_files.npy")
    print(f"[DONE] Computed embeddings for {len(embs)} files.")
else:
    print("[WARN] No embeddings computed. Check input files.")

if errors:
    print(f"[SUMMARY] {len(errors)} files failed to process:")
    for fname, err in errors:
        print(f"  - {fname}: {err}")
else:
    print("[SUMMARY] All files processed successfully.")

print(f"[INFO] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
