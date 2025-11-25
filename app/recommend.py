import numpy as np

def recommend(
    query_emb: np.ndarray,
    lib_embs: np.ndarray,
    lib_meta,
    top_k: int = 10,
):
    """
    FMA-only recommendation using cosine similarity on embeddings.

    query_emb: (d_emb,)
    lib_embs:  (N, d_emb)
    lib_meta:  array/list of meta dicts with at least:
               - "track_id"
               - "filename"
               Optionally: "artist", "title"
    """
    # --- Cosine similarity on embeddings ---
    eps = 1e-8
    query_emb = query_emb / (np.linalg.norm(query_emb) + eps)
    lib_embs_norm = lib_embs / (np.linalg.norm(lib_embs, axis=1, keepdims=True) + eps)

    sim = lib_embs_norm @ query_emb  # (N,)

    # --- Top-k indices ---
    idx = np.argsort(-sim)[:top_k]

    recs = []
    for i in idx:
        # lib_meta may be a numpy array with dtype=object, so unwrap if needed
        m = lib_meta[i].item() if not isinstance(lib_meta[i], dict) else lib_meta[i]

        title = (
            m.get("title")
            or m.get("filename")
            or m.get("track_id")
            or "Unknown track"
        )
        artist = m.get("artist", "Unknown artist")

        recs.append(
            {
                "title": title,
                "artist": artist,
                "similarity": float(sim[i]),
                "spotify_id": m.get("spotify_id"),
                "spotify_url": m.get("spotify_url"),
                "meta": m,
            }
        )

        
    return recs
