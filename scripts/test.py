import os
from pathlib import Path
import sys, os
# Add project root to path dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.spotify_utils import get_spotify_client

def main():
    sp = get_spotify_client()
    results = sp.search(q="Beatles", type="track", limit=3)
    for t in results["tracks"]["items"]:
        print(t["name"], "-", t["artists"][0]["name"])

if __name__ == "__main__":
    main()
