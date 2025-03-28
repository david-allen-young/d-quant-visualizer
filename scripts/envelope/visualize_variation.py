import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# Load generated envelopes
data_dir = "./viz_batch"
files = sorted(glob.glob(os.path.join(data_dir, "generated_morph2_*.csv")))

envelopes = []

for file in files:
    df = pd.read_csv(file)
    if "Expression" in df.columns and len(df["Expression"]) == 100:
        envelopes.append(df["Expression"].values)
    else:
        print(f"Skipping malformed or empty file: {file}")

if len(envelopes) == 0:
    raise ValueError("No valid envelope data found. Check your input folder and file format.")

X = np.array(envelopes)  # Shape: (n_samples, 100)


# === PCA Projection ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k')
plt.title("PCA Projection of Morph2 Envelopes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Distance Heatmap ===
dist_matrix = squareform(pdist(X, metric="euclidean"))

plt.figure(figsize=(10, 8))
plt.imshow(dist_matrix, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Euclidean Distance")
plt.title("Similarity Heatmap of Generated Envelopes")
plt.xlabel("Envelope Index")
plt.ylabel("Envelope Index")
plt.tight_layout()
plt.show()
