import os
import requests
import spotipy

from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()   # This loads .env automatically

_SP_CLIENT = None


def get_spotify_client():
    global _SP_CLIENT
    if _SP_CLIENT is None:
        client_id = os.environ.get("SPOTIFY_CLIENT_ID")
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set as environment variables."
            )
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        _SP_CLIENT = spotipy.Spotify(auth_manager=auth)
    return _SP_CLIENT


def fetch_playlist_tracks(playlist_id: str, limit=100):
    """
    Returns a list of dicts with keys: id, name, artist, url, preview_url

    NOTE: This will fail (404) for many Spotify-owned/editorial playlists.
    Kept here in case you want to use user-owned playlists elsewhere.
    """
    sp = get_spotify_client()
    items = sp.playlist_items(playlist_id, limit=limit)["items"]
    tracks = []
    for it in items:
        t = it.get("track")
        if not t:
            continue
        tracks.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "url": t["external_urls"]["spotify"],
            "preview_url": t["preview_url"],
        })
    return tracks


def fetch_audio_features(track_id: str):
    """
    Returns an 8-D feature vector using Spotify's audio_features endpoint,
    or None if not available.
    """
    sp = get_spotify_client()
    feats = sp.audio_features([track_id])[0]
    if feats is None:
        return None

    # Selected features
    return [
        feats["danceability"],
        feats["energy"],
        feats["speechiness"],
        feats["acousticness"],
        feats["instrumentalness"],
        feats["liveness"],
        feats["valence"],
        feats["tempo"],
    ]


def fetch_global_weekly_with_previews(max_tracks: int = 100, oversample: int = 300):
    """
    Fetch up to `oversample` entries from the global weekly chart,
    then use Web API to keep only tracks that have a non-null preview_url.

    Returns a list of dicts with keys: id, name, artist, url, preview_url.
    """
    # Step 1: get chart entries (same as before but up to `oversample`)
    url = "https://charts-spotify-com-service.spotify.com/public/v0/charts"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    charts = data.get("chartEntryViewResponses", [])
    if not charts:
        return []

    global_chart = None
    for c in charts:
        cid = c.get("id", "") or c.get("chart", {}).get("id", "")
        name = c.get("name", "") or c.get("chart", {}).get("name", "")
        if "top200" in cid and "global" in cid:
            global_chart = c
            break
        if "Global Top 200" in name:
            global_chart = c
            break

    if global_chart is None:
        global_chart = charts[0]

    entries = global_chart.get("entries", [])[:oversample]

    # Step 2: look up track details in batches and keep only those with preview_url
    sp = get_spotify_client()
    track_ids = []
    meta_by_id = {}

    for entry in entries:
        meta = entry.get("trackMetadata", {})
        uri = (
            meta.get("trackUri")
            or meta.get("trackUriSpotify")
            or meta.get("uri")
        )
        if not uri:
            continue
        tid = uri.split(":")[-1]
        track_ids.append(tid)
        meta_by_id[tid] = meta

    tracks = []
    # Spotify tracks endpoint allows up to 50 IDs per call
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i+50]
        resp = sp.tracks(batch)
        for t in resp["tracks"]:
            if t is None:
                continue
            tid = t["id"]
            preview_url = t.get("preview_url")
            if not preview_url:
                continue  # skip tracks without preview

            meta = meta_by_id.get(tid, {})
            artists_list = meta.get("artists", []) or t.get("artists", [])
            artist_name = artists_list[0]["name"] if artists_list else "Unknown artist"

            tracks.append({
                "id": tid,
                "name": meta.get("trackName", t.get("name", "Unknown track")),
                "artist": artist_name,
                "url": t.get("external_urls", {}).get("spotify", f"https://open.spotify.com/track/{tid}"),
                "preview_url": preview_url,
            })

            if len(tracks) >= max_tracks:
                return tracks

    return tracks

