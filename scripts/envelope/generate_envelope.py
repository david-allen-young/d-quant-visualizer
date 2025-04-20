import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================================
# Usage:
#
# Analyze envelope CSVs and generate mean/std:
# python generate_envelope.py analyze --csv_dir ... --output_dir ...
#
# Generate new envelopes using morph2 or morph2_buckets:
# python generate_envelope.py generate --method morph2_buckets --mean_path ... --std_path ... --input_csv_dir ... --count 100 --save_dir ... --bucket_map_path ... --category stable
# ============================================

TARGET_LENGTH = 100

def analyze(csv_dir, output_dir):
    file_list = sorted(glob.glob(os.path.join(csv_dir, "envelope*.csv")))
    print("Found files:", file_list)

    interpolated_envelopes = []

    for file in file_list:
        df = pd.read_csv(file)
        orig_x = df["Position"].values
        orig_y = df["Expression"].values

        interp_x = np.linspace(orig_x.min(), orig_x.max(), TARGET_LENGTH)
        interp_y = np.interp(interp_x, orig_x, orig_y)
        interpolated_envelopes.append(interp_y)

    envelope_matrix = np.array(interpolated_envelopes)
    mean_envelope = envelope_matrix.mean(axis=0)
    std_envelope = envelope_matrix.std(axis=0)
    x_vals = np.linspace(0, 1, TARGET_LENGTH)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mean_envelope.npy"), mean_envelope)
    np.save(os.path.join(output_dir, "std_envelope.npy"), std_envelope)

    # Plot
    plt.figure(figsize=(12, 6))
    for y_values in envelope_matrix:
        plt.plot(x_vals, y_values, alpha=0.3, linewidth=1)

    plt.fill_between(x_vals, mean_envelope - std_envelope, mean_envelope + std_envelope,
                     color='gray', alpha=0.3, label='+/-1 Std Dev')
    plt.plot(x_vals, mean_envelope, color='black', linestyle='--', linewidth=2, label='Mean Envelope')

    plt.xlabel("Normalized Position (0-1)")
    plt.ylabel("Expression (0-1)")
    plt.title("Overlay of MIDI Expression Envelopes with Mean and Std Dev")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def morph_from_two_inputs(mean, input1, input2, morph_factor1=0.5, morph_factor2=0.5, std=None, category="crescendo"):
    trend = 0.5 * (input1 + input2)
    delta = trend - mean
    morph_factor = 0.5 * (morph_factor1 + morph_factor2)
    adaptive_morph = morph_factor * (std / (std.max() + 1e-8))
    morphed = mean + adaptive_morph * delta
    if category != "stable":
        morphed = mean + soft_clip(morphed - mean, std)
    return np.clip(morphed, 0, 1)

def soft_clip(delta, std, softness=3.0):
    return std * np.tanh(delta / (std + 1e-8) * softness)

def center_envelope(env):
    return env - np.mean(env)

def generate(mean_path, std_path, method="noise", count=5, strength=0.8, seed=None, save_dir=None, input_csv_dir=None, category="crescendo", bucket_map_path=None):
    mean = np.load(mean_path)
    std = np.load(std_path)
    x_vals = np.linspace(0, 1, TARGET_LENGTH)

    if seed is not None:
        np.random.seed(seed)

    file_list = sorted(glob.glob(os.path.join(input_csv_dir, "envelope*.csv")))
    envelope_data = {}

    for file in file_list:
        df = pd.read_csv(file)
        orig_x = df["Position"].values
        orig_y = df["Expression"].values
        interp_y = np.interp(np.linspace(orig_x.min(), orig_x.max(), TARGET_LENGTH), orig_x, orig_y)
        if category == "stable":
            interp_y = center_envelope(interp_y)
        envelope_data[os.path.basename(file)] = interp_y

    if method == "morph2_buckets":
        with open(bucket_map_path) as f:
            bucket_map = json.load(f)

        # Group by cluster
        clusters = {}
        for fname, cluster_id in bucket_map.items():
            clusters.setdefault(cluster_id, []).append(envelope_data[fname])

        for cluster_id, cluster_envs in clusters.items():
            plt.figure(figsize=(12, 6))
            for _ in range(min(count, len(cluster_envs))):
                i1, i2 = np.random.choice(len(cluster_envs), size=2, replace=False)
                f1, f2 = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
                env = morph_from_two_inputs(mean, cluster_envs[i1], cluster_envs[i2], f1, f2, std=std, category=category)
                plt.plot(x_vals, env, alpha=0.3)
            plt.title(f"Overlay: Stable Envelope Cluster {cluster_id}")
            plt.xlabel("Normalized Position")
            plt.ylabel("Expression")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    else:
        # Existing logic (morph, morph2, noise)
        original_inputs = list(envelope_data.values())
        plt.figure(figsize=(12, 6))
        for i in range(count):
            if method == "morph2":
                while True:
                    i1, i2 = np.random.choice(len(original_inputs), size=2, replace=False)
                    d1 = np.sign(original_inputs[i1] - mean)
                    d2 = np.sign(original_inputs[i2] - mean)
                    if np.mean(d1 == d2) > 0.8:
                        break
                f1, f2 = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
                env = morph_from_two_inputs(mean, original_inputs[i1], original_inputs[i2], f1, f2, std=std, category=category)
                plt.plot(x_vals, env, alpha=0.7)

        plt.plot(x_vals, mean, color='black', linestyle='--', linewidth=2, label='Mean')
        plt.fill_between(x_vals, mean - std, mean + std, color='gray', alpha=0.3)
        plt.title(f"Generated Envelopes using '{method}' method")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze or generate MIDI expression envelopes.")
    parser.add_argument("mode", choices=["analyze", "generate"], help="Mode: analyze or generate.")
    parser.add_argument("--method", choices=["noise", "morph", "morph2", "morph2_buckets"], default="noise", help="Generation method.")
    parser.add_argument("--csv_dir", type=str, default="./sample_data", help="Input directory for CSV files.")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Where to save analysis results.")
    parser.add_argument("--mean_path", type=str, help="Path to mean_envelope.npy for generate mode.")
    parser.add_argument("--std_path", type=str, help="Path to std_envelope.npy for generate mode.")
    parser.add_argument("--input_csv_dir", type=str, help="Path to original input envelopes (required for morph modes).")
    parser.add_argument("--count", type=int, default=5, help="Number of envelopes to generate.")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength of deviation (0-1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, help="Optional output directory for generated envelopes.")
    parser.add_argument("--category", type=str, default="crescendo", help="Data category: crescendo, diminuendo, or stable")
    parser.add_argument("--bucket_map_path", type=str, default=None, help="Optional: path to bucket_map.json for morph2_buckets")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze(args.csv_dir, args.output_dir)
    elif args.mode == "generate":
        if not args.mean_path or not args.std_path:
            print("Please specify --mean_path and --std_path for generate mode.")
        else:
            generate(args.mean_path, args.std_path, method=args.method, count=args.count,
                     strength=args.strength, seed=args.seed, save_dir=args.save_dir,
                     input_csv_dir=args.input_csv_dir, category=args.category,
                     bucket_map_path=args.bucket_map_path)

if __name__ == "__main__":
    main()
