import sys, os
from urllib.parse import quote_plus
import pandas as pd
import streamlit.components.v1 as components

# === FIX sys.path so we can import app.*
ROOT = os.path.dirname(os.path.abspath(__file__))   # .../song_project/app
ROOT = os.path.dirname(ROOT)                        # .../song_project
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np

from app.configs import FMA_EMBS_PATH, FMA_META_PATH
from app.audio_utils import local_file_to_embedding
from app.recommend import recommend

# === UI HEADER ===
st.set_page_config(page_title="Neural Music Recommender", page_icon="🎧")
st.title("🎧 Neural Music Recommender")
st.write("Upload a song and get similar tracks from our audio-embedded library.")

# === Load precomputed FMA embedding library ===
@st.cache_resource
def load_library():
    embs = np.load(FMA_EMBS_PATH)
    meta = np.load(FMA_META_PATH, allow_pickle=True)
    return embs, meta

lib_embs, lib_meta = load_library()

# === Upload UI ===
uploaded = st.file_uploader("Upload an MP3 file", type=["mp3"])

# =============== MAIN LOGIC ===============
if uploaded is not None:
    st.audio(uploaded, format="audio/mp3")

    # Compute embedding
    with st.spinner("Computing embedding..."):
        try:
            query_emb = local_file_to_embedding(uploaded)
        except Exception as e:
            st.error(f"Could not process audio: {e}")
            st.stop()

    # Get recommendations
    with st.spinner("Finding similar tracks..."):
        recs = recommend(
            query_emb=query_emb,
            lib_embs=lib_embs,
            lib_meta=lib_meta,
            top_k=10,
        )

    # ========== DISPLAY RECOMMENDATIONS ==========
    st.subheader("Recommendations")

    top_viz = recs[:5]  # For the similarity bar chart

    for r in recs:
        title = r["title"]
        artist = r["artist"]
        sim = r["similarity"]
        spotify_id = r.get("spotify_id")
        spotify_url = r.get("spotify_url")

        st.markdown(f"**{title}** – {artist}  \nSimilarity: `{sim:.3f}`")

        # Spotify embed if available
        if spotify_id:
            embed_url = f"https://open.spotify.com/embed/track/{spotify_id}"
            components.iframe(embed_url, height=80)
        elif spotify_url:
            st.markdown(f"[Open in Spotify]({spotify_url})")
        else:
            search_url = f"https://open.spotify.com/search/{quote_plus(title + ' ' + artist)}"
            st.markdown(f"[Search on Spotify]({search_url})")

        st.markdown("---")

    # === Similarity chart for top 5 ===
    if top_viz:
        st.subheader("Similarity (top 5)")
        df = pd.DataFrame({
            "Track": [f"{r['title']} – {r['artist']}" for r in top_viz],
            "Similarity": [r["similarity"] for r in top_viz],
        }).set_index("Track")
        st.bar_chart(df)

# =============== NO FILE UPLOADED ===============
else:
    st.info("Upload an MP3 file to get started.")
