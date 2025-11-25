
from pathlib import Path
import sys, os
# Add project root to path dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

from app.spotify_utils import fetch_global_weekly_with_previews, fetch_audio_features
from app.audio_utils import mp3_bytes_to_mel_chunks
from app.embedding_utils import embed_mel_chunks
from app.configs import DATA_DIR

import requests


OUT_EMB = DATA_DIR / "spotify_embs.npy"
OUT_FEATS = DATA_DIR / "spotify_feats_scaled.npy"
OUT_META = DATA_DIR / "spotify_meta.npy"
OUT_SCALER = DATA_DIR / "scaler.pkl"

DATA_DIR.mkdir(exist_ok=True, parents=True)

def main():

    tracks = fetch_global_weekly_with_previews(max_tracks=100, oversample=5000)
    print(f"Got {len(tracks)} tracks with preview URLs.")
    if not tracks:
        print("No tracks with previews found from chart.")
        return

    embs = []
    feats = []
    meta = []

    for t in tracks:
        if t["preview_url"] is None:
            print("No preview for:", t["name"], "-", t["artist"])
            continue

        try:
            r = requests.get(t["preview_url"], timeout=10)
            r.raise_for_status()
        except Exception as e:
            print("Error downloading preview for", t["name"], ":", e)
            continue

        mel_chunks = mp3_bytes_to_mel_chunks(r.content)
        if mel_chunks is None:
            print("Too short / bad audio:", t["name"], "-", t["artist"])
            continue

        emb = embed_mel_chunks(mel_chunks)
        fvec = fetch_audio_features(t["id"])
        if fvec is None:
            print("No audio features for:", t["name"], "-", t["artist"])
            continue

        embs.append(emb)
        feats.append(fvec)
        meta.append(t)

    if not embs:
        print("No tracks successfully embedded.")
        return

    embs = np.stack(embs)   # (N, d_emb)
    feats = np.stack(feats) # (N, d_feat)
    meta = np.array(meta, dtype=object)

    # Scale features
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    np.save(OUT_EMB, embs)
    np.save(OUT_FEATS, feats_scaled)
    np.save(OUT_META, meta)

    with open(OUT_SCALER, "wb") as f:
        pickle.dump(scaler, f)

    print("Library built with", embs.shape[0], "tracks.")

if __name__ == "__main__":
    main()
