# cluster_stable_envelopes.py
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN  # optional for later

TARGET_LENGTH = 100

def center_envelope(arr):
    return arr - np.mean(arr)

def load_and_preprocess_envelopes(csv_dir, center=True):
    file_list = sorted(glob.glob(os.path.join(csv_dir, "envelope*.csv")))
    names, matrix = [], []
    for f in file_list:
        df = pd.read_csv(f)
        x = df["Position"].values
        y = df["Expression"].values
        y_interp = np.interp(np.linspace(x.min(), x.max(), TARGET_LENGTH), x, y)
        if center:
            y_interp = center_envelope(y_interp)
        matrix.append(y_interp)
        names.append(os.path.basename(f))
    return names, np.array(matrix)

def run_pca(matrix, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(matrix)
    return reduced, pca

def run_clustering(data_2d, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data_2d)
    return labels

def save_cluster_map(names, labels, out_path):
    mapping = dict(zip(names, labels.tolist()))
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Saved cluster mapping to: {out_path}")

def visualize_clusters(data_2d, labels, names, out_file="pca_clusters.png"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", s=50)
    plt.title("PCA Projection of Envelope Shapes with Cluster Labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"[INFO] Saved PCA visualization to: {out_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory with envelope CSVs")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters for KMeans")
    parser.add_argument("--output_json", type=str, default="bucket_map.json", help="Path to save cluster map")
    parser.add_argument("--output_plot", type=str, default="pca_clusters.png", help="Path to save PCA plot")
    args = parser.parse_args()

    names, matrix = load_and_preprocess_envelopes(args.csv_dir, center=True)
    data_2d, _ = run_pca(matrix)
    labels = run_clustering(data_2d, args.n_clusters)
    save_cluster_map(names, labels, args.output_json)
    visualize_clusters(data_2d, labels, names, args.output_plot)

if __name__ == "__main__":
    main()
