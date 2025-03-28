import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# Load generated envelopes
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "viz_batch")
files = sorted(glob.glob(os.path.join(data_dir, "generated_morph2_*.csv")))

envelopes = []

for file in files:
    print(f"Reading {file}")
    df = pd.read_csv(file)
    print(f"Columns: {df.columns}")
    print(f"Length: {len(df)}")

    if "Expression" in df.columns:
        expr = df["Expression"].values[:100]  # safely trim to 100 samples
        if len(expr) == 100:
            envelopes.append(expr)
        else:
            print(f"Error: Skipping short envelope in {file}")
    else:
        print(f"Error: 'Expression' column not found in {file}")


if len(envelopes) == 0:
    raise ValueError("No valid envelope data found. Check your input folder and file format.")

X = np.array(envelopes)  # Shape: (n_samples, 100)


# === PCA Projection ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Compute distance from the PCA origin (0, 0)
pca_distances = np.linalg.norm(X_pca, axis=1)


plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pca_distances, cmap='viridis', edgecolors='k', alpha=0.8)
plt.colorbar(scatter, label="Distance from Mean (PCA space)")
plt.title("PCA Projection of Morph2 Envelopes (Colored by Deviation)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()


# # === Distance Heatmap ===
# dist_matrix = squareform(pdist(X, metric="euclidean"))

# plt.figure(figsize=(10, 8))
# plt.imshow(dist_matrix, cmap="viridis", interpolation="nearest")
# plt.colorbar(label="Euclidean Distance")
# plt.title("Similarity Heatmap of Generated Envelopes")
# plt.xlabel("Envelope Index")
# plt.ylabel("Envelope Index")
# plt.tight_layout()
# plt.show()

# === Envelope Overlay Plot (like generator's visualization) ===
x_vals = np.linspace(0, 1, X.shape[1])
mean = X.mean(axis=0)
std = X.std(axis=0)

plt.figure(figsize=(12, 6))
for y in X:
    plt.plot(x_vals, y, alpha=0.3, linewidth=1)

plt.fill_between(x_vals, mean - std, mean + std, color='gray', alpha=0.3, label='+/-1 Std Dev')
plt.plot(x_vals, mean, color='black', linestyle='--', linewidth=2, label='Mean')

plt.title("Overlay of Generated Envelopes")
plt.xlabel("Normalized Position (0-1)")
plt.ylabel("Expression (0-1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
