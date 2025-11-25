from pathlib import Path
import sys, os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

import pandas as pd


# Add project root to path dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding_utils import embed_mel_chunks
from app.spotify_utils import get_spotify_client
from app.configs import DATA_DIR

# ====== CONFIG ======
# Folder where your FMA mel-spectrogram .npz files live.
# Adjust this if yours is different.
MEL_DIR = DATA_DIR / "spectrograms"   # e.g. data/fma_mels/*.npz

CHUNK_LENGTH = 129      # same as SpectrogramDataset
N_MELS = 128            # for reference; not strictly needed here

OUT_EMB = DATA_DIR / "fma_embs.npy"
OUT_META = DATA_DIR / "fma_meta.npy"
# Optional if you later add extra features:
OUT_FEATS = DATA_DIR / "fma_feats_scaled.npy"
OUT_SCALER = DATA_DIR / "fma_scaler.pkl"

DATA_DIR.mkdir(exist_ok=True, parents=True)

# Path like: data/fma_metadata/tracks.csv
TRACKS_CSV = DATA_DIR / "fma_metadata" / "fma_metadata" / "tracks.csv"

print("Loading FMA track metadata from:", TRACKS_CSV)
tracks_df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

# Convenience columns (FMA uses multi-index columns)
artist_col = ("artist", "name")
title_col = ("track", "title")



def chunk_spectrogram(mel_db: np.ndarray, chunk_length: int = CHUNK_LENGTH):
    """
    Divide mel spectrogram (n_mels, n_frames) into time chunks of equal length.
    Same logic as SpectrogramDataset.chunk_spectrogram.
    """
    n_frames = mel_db.shape[1]
    chunks = []
    for start in range(0, n_frames, chunk_length):
        end = start + chunk_length
        if end <= n_frames:
            chunks.append(mel_db[:, start:end])
    if not chunks:
        return None
    return np.stack(chunks)  # (n_chunks, n_mels, chunk_length)


def load_mel_chunks_from_npz(path: Path):
    """
    Load mel-spectrogram from an .npz file and return time chunks.

    - Assumes the file has a key "mel".
    - If it's 2D (n_mels, n_frames), we chunk it.
    - If it's already 3D, we assume it's (n_chunks, n_mels, chunk_length).
    """
    data = np.load(path)
    if "mel" not in data:
        print(f"Skipping {path.name}: no 'mel' key in npz.")
        return None

    mel = data["mel"]

    # 2D: chunk on the fly (same as in SpectrogramDataset)
    if mel.ndim == 2:
        chunks = chunk_spectrogram(mel)
        return chunks

    # 3D: assume already chunked correctly
    if mel.ndim == 3:
        return mel

    print(f"Skipping {path.name}: unexpected mel ndim={mel.ndim}")
    return None

def find_spotify_track_id(sp, artist: str, title: str) -> str | None:
    """
    Try to find a Spotify track ID using artist + title.
    Returns None if not found.
    """
    query = f"track:{title} artist:{artist}"
    try:
        results = sp.search(q=query, type="track", limit=1)
    except Exception as e:
        print("Spotify search error:", e)
        return None

    items = results.get("tracks", {}).get("items", [])
    if not items:
        return None
    return items[0]["id"]



def main():
    npz_paths = sorted(MEL_DIR.glob("*.npz"))
    print(f"Found {len(npz_paths)} FMA spectrogram files in {MEL_DIR}.")

    embs = []
    meta = []
    feats = []   # optional, if you later compute extra features


    sp = get_spotify_client()


    for i, path in enumerate(npz_paths, 1):
        if i % 100 == 0:
            print(f"{i}/{len(npz_paths)} processed...")

        mel_chunks = load_mel_chunks_from_npz(path)
        if mel_chunks is None:
            print("Skipping (no valid mel chunks):", path.name)
            continue

        try:
            # embed_mel_chunks should accept an array of shape
            # (n_chunks, n_mels, chunk_length) and return a single embedding
            emb = embed_mel_chunks(mel_chunks)
        except Exception as e:
            print("Error embedding", path.name, ":", e)
            continue

        embs.append(emb)

        # ---- META ----
        track_id = path.stem  # e.g. "000123" from "000123.npz"
                # ---- META ----
        track_id_str = path.stem         # e.g. "012537"
        track_id = int(track_id_str)     # FMA IDs are ints

        # Look up artist / title from tracks.csv
        if track_id in tracks_df.index:
            row = tracks_df.loc[track_id]
            artist = row[artist_col]
            title = row[title_col]
        else:
            artist = "Unknown artist"
            title = track_id_str

        # Try to find Spotify ID (optional, but you want it)
        spotify_id = find_spotify_track_id(sp, artist, title)
        spotify_url = (
            f"https://open.spotify.com/track/{spotify_id}"
            if spotify_id
            else None
        )

        meta.append(
            {
                "track_id": track_id,
                "filename": path.name,
                "artist": artist,
                "title": title,
                "spotify_id": spotify_id,
                "spotify_url": spotify_url,
            }
        )


        # ---- Optional: extra features ----
        # feats.append(...)

    if not embs:
        print("No tracks successfully embedded from FMA.")
        return

    embs = np.stack(embs)  # (N, d_emb)
    meta = np.array(meta, dtype=object)

    print("Built embeddings for", embs.shape[0], "tracks.")

    # Save optional extra features + scaler if you use them
    if feats:
        feats = np.stack(feats)
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats)

        np.save(OUT_FEATS, feats_scaled)
        with open(OUT_SCALER, "wb") as f:
            pickle.dump(scaler, f)
        print("Saved extra features with scaler to:", OUT_FEATS, "and", OUT_SCALER)

    np.save(OUT_EMB, embs)
    np.save(OUT_META, meta)

    print("FMA library built and saved to:")
    print("  Embeddings:", OUT_EMB)
    print("  Metadata:  ", OUT_META)


if __name__ == "__main__":
    main()
