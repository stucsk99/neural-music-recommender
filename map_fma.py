import pandas as pd
import numpy as np
import os


SPEC_DIR = "E:/song_project/spectrograms"
EMB_DIR  = "E:/song_project"

meta = pd.read_csv("E:/song_project/data/fma_metadata/fma_metadata/tracks.csv",
                   header=[0,1], index_col=0)
genre_map = meta[("track", "genre_top")]

spec_files = sorted([f for f in os.listdir(SPEC_DIR) if f.endswith(".npz")])

genres = []
valid_files = []

for f in spec_files:
    track_id = int(f.replace(".npz", ""))    # extract 123456 from "123456.npz"
    if track_id in genre_map.index:
        genres.append(genre_map.loc[track_id])
        valid_files.append(f)
    else:
        print("Warning: no metadata for", f)

genres = np.array(genres)
files  = np.array(valid_files)

# Save
np.save(f"{EMB_DIR}/genre_labels.npy", genres)
np.save(f"{EMB_DIR}/spectrogram_files_used.npy", files)

print("Saved", len(genres), "genre labels.")
