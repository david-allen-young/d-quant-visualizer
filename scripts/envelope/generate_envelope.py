import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Parameters
target_length = 100


def analyze(csv_dir, output_dir):
    file_list = sorted(glob.glob(os.path.join(csv_dir, "envelope*.csv")))
    # print("Found files:", file_list)

    interpolated_envelopes = []

    for file in file_list:
        df = pd.read_csv(file)
        orig_x = df["Position"].values
        orig_y = df["Expression"].values

        interp_x = np.linspace(orig_x.min(), orig_x.max(), target_length)
        interp_y = np.interp(interp_x, orig_x, orig_y)
        interpolated_envelopes.append(interp_y)

    envelope_matrix = np.array(interpolated_envelopes)
    mean_envelope = envelope_matrix.mean(axis=0)
    std_envelope = envelope_matrix.std(axis=0)
    x_vals = np.linspace(0, 1, target_length)

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


def generate(mean_path, std_path, count=5, strength=0.8, seed=None, save_dir=None):
    mean = np.load(mean_path)
    std = np.load(std_path)
    x_vals = np.linspace(0, 1, target_length)

    if seed is not None:
        np.random.seed(seed)

    plt.figure(figsize=(12, 6))
    for i in range(count):
        noise = np.random.randn(len(mean))
        smooth_noise = np.convolve(noise, np.ones(10) / 10, mode='same')
        smooth_noise /= np.max(np.abs(smooth_noise))

        envelope = mean + smooth_noise * std * strength
        envelope = np.clip(envelope, 0, 1)
        plt.plot(x_vals, envelope, label=f'Generated {i+1}', alpha=0.7)

        # Optional save
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_df = pd.DataFrame({"Position": x_vals, "Expression": envelope})
            out_df.to_csv(os.path.join(save_dir, f"generated_envelope_{i+1}.csv"), index=False)

    plt.plot(x_vals, mean, color='black', linestyle='--', linewidth=2, label='Mean')
    plt.fill_between(x_vals, mean - std, mean + std, color='gray', alpha=0.3)
    plt.title("Generated Envelopes within +/-1 Std Dev")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze or generate MIDI expression envelopes.")
    parser.add_argument("mode", choices=["analyze", "generate"], help="Mode: analyze or generate.")
    parser.add_argument("--csv_dir", type=str, default="./sample_data", help="Input directory for CSV files.")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Where to save analysis results.")
    parser.add_argument("--mean_path", type=str, help="Path to mean_envelope.npy for generate mode.")
    parser.add_argument("--std_path", type=str, help="Path to std_envelope.npy for generate mode.")
    parser.add_argument("--count", type=int, default=5, help="Number of envelopes to generate.")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength of deviation (0-1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--save_dir", type=str, help="Optional output directory for generated envelopes.")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze(args.csv_dir, args.output_dir)
    elif args.mode == "generate":
        if not args.mean_path or not args.std_path:
            print("Please specify --mean_path and --std_path for generate mode.")
        else:
            generate(args.mean_path, args.std_path, args.count, args.strength, args.seed, args.save_dir)


if __name__ == "__main__":
    main()
