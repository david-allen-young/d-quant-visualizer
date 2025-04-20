import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# ============================================
# Usage:
#
# Analyze envelope CSVs and generate mean/std:
# python generate_envelope.py analyze --csv_dir "../../../d-quant/assets/output_csv/" --output_dir "./analysis"
#
# Generate new envelopes using morph2 (2-input blend with soft clamping):
# python generate_envelope.py generate --method morph2 --mean_path ./analysis/mean_envelope.npy --std_path ./analysis/std_envelope.npy --input_csv_dir "../../../d-quant/assets/output_csv/" --count 100 --save_dir ./viz_batch
#
# ============================================


# Parameters
target_length = 100

def analyze(csv_dir, output_dir):
    file_list = sorted(glob.glob(os.path.join(csv_dir, "envelope*.csv")))
    print("Found files:", file_list)

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

def morph_toward_input(mean, target_input, morph_factor=0.5, std=None, clamp_to_std=True):
    delta = target_input - mean
    adaptive_morph = morph_factor * (std / (std.max() + 1e-8))
    morphed = mean + adaptive_morph * delta
    if clamp_to_std and std is not None:
        bounded_delta = soft_clip(morphed - mean, std)
        morphed = mean + bounded_delta
    return np.clip(morphed, 0, 1)

def morph_from_two_inputs(mean, input1, input2, morph_factor1=0.5, morph_factor2=0.5, std=None):
    trend = 0.5 * (input1 + input2)
    delta = trend - mean
    morph_factor = 0.5 * (morph_factor1 + morph_factor2)
    adaptive_morph = morph_factor * (std / (std.max() + 1e-8))
    morphed = mean + adaptive_morph * delta
    soft_clamped = soft_clip(morphed - mean, std)
    return np.clip(mean + soft_clamped, 0, 1)

def soft_clip(delta, std, softness=3.0):
    return std * np.tanh(delta / (std + 1e-8) * softness)

def center_envelope(env):
    mean = np.mean(env)
    return env - mean

def generate(mean_path, std_path, method="noise", count=5, strength=0.8, seed=None, save_dir=None, input_csv_dir=None, category="crescendo"):
    mean = np.load(mean_path)
    std = np.load(std_path)
    x_vals = np.linspace(0, 1, target_length)

    if seed is not None:
        np.random.seed(seed)

    if method in ["morph", "morph2"]:
        if not input_csv_dir:
            raise ValueError("--input_csv_dir must be specified when using 'morph' or 'morph2' method.")
        file_list = sorted(glob.glob(os.path.join(input_csv_dir, "envelope*.csv")))
        original_inputs = []

        for file in file_list:
            df = pd.read_csv(file)
            orig_x = df["Position"].values
            orig_y = df["Expression"].values
            interp_y = np.interp(np.linspace(orig_x.min(), orig_x.max(), target_length), orig_x, orig_y)
            if category == "stable":
                centered_y = center_envelope(interp_y)
                original_inputs.append(centered_y)
            else:
                original_inputs.append(interp_y)

    plt.figure(figsize=(12, 6))
    for i in range(count):
        if method == "morph":
            target = original_inputs[np.random.randint(len(original_inputs))]
            morph_factor = np.random.uniform(0.2, 0.8)
            envelope = morph_toward_input(mean, target, morph_factor, std=std, clamp_to_std=True)

        elif method == "morph2":
            while True:
                i1, i2 = np.random.choice(len(original_inputs), size=2, replace=False)
                d1 = np.sign(original_inputs[i1] - mean)
                d2 = np.sign(original_inputs[i2] - mean)
                agreement = np.mean(d1 == d2)
                if agreement > 0.8:
                    break
            f1, f2 = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
            envelope = morph_from_two_inputs(mean, original_inputs[i1], original_inputs[i2], f1, f2, std=std)

        else:  # noise-based
            noise = np.random.randn(len(mean))
            smooth_noise = np.convolve(noise, np.ones(10) / 10, mode='same')
            smooth_noise /= np.max(np.abs(smooth_noise))
            envelope = mean + smooth_noise * std * strength
            envelope = np.clip(envelope, 0, 1)

        plt.plot(x_vals, envelope, label=f'{method.capitalize()} {i+1}', alpha=0.7)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            df = pd.DataFrame({"Position": x_vals, "Expression": envelope})
            df.to_csv(os.path.join(save_dir, f"generated_{method}_{i+1}.csv"), index=False)

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
    parser.add_argument("--method", choices=["noise", "morph", "morph2"], default="noise", help="Generation method.")
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
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze(args.csv_dir, args.output_dir)
    elif args.mode == "generate":
        if not args.mean_path or not args.std_path:
            print("Please specify --mean_path and --std_path for generate mode.")
        else:
            generate(args.mean_path, args.std_path, method=args.method, count=args.count,
                     strength=args.strength, seed=args.seed, save_dir=args.save_dir,
                     input_csv_dir=args.input_csv_dir, category=args.category)

if __name__ == "__main__":
    main()
