import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X = np.load("E:/song_project/embeddings.npy")   # (n_songs, embed_dim)
# Optional: if you have integer genre labels array the same length:
# y = np.load("E:/song_project/labels.npy")
genres = np.load("E:/song_project/genre_labels.npy")

# Convert genres → integers for coloring
unique = np.unique(genres)
genre_to_int = {g:i for i,g in enumerate(unique)}
y = np.array([genre_to_int[g] for g in genres])

# PCA (fast, gives a quick look)
pca = PCA(n_components=2)
p2 = pca.fit_transform(X)



plt.figure(figsize=(6,5))
plt.scatter(p2[:,0], p2[:,1], c=y, cmap='tab10', s=10, alpha=0.7)  # , c=y, cmap='tab10'
plt.title("PCA of song embeddings")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("E:/song_project/pca_embeddings.png", dpi=150, bbox_inches='tight')
print("[INFO] Saved PCA plot to E:/song_project/pca_embeddings.png")
plt.close()

# t-SNE (slower, often prettier local clusters)
ts = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
t2 = ts.fit_transform(X)

plt.figure(figsize=(6,5))
plt.scatter(t2[:,0], t2[:,1],  c=y, cmap='tab10', s=10, alpha=0.7)  # , c=y, cmap='tab10'
plt.title("t-SNE of song embeddings"); plt.tight_layout()
plt.savefig("E:/song_project/tsne_embeddings.png", dpi=150, bbox_inches='tight')
print("[INFO] Saved t-SNE plot to E:/song_project/tsne_embeddings.png")
plt.close()

print("[DONE] Visualization complete. Open the PNG files in your file explorer or image viewer.")
