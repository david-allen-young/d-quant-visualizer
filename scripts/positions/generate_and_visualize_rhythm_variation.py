import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# def soft_clip_scalar(delta, std, softness=3.0):
#     return std * np.tanh((delta / (std + 1e-8)) * softness)

# def generate_rhythm_sample(mean, std, softness=3.0):
#     delta = np.random.randn()
#     return mean + soft_clip_scalar(delta, std, softness)

def soft_clip_scalar(delta, std, softness=1.0):
    """
    Softly clamps the deviation using tanh, scaled by std and a configurable softness factor.
    Lower softness = gentler clamp, Higher softness = tighter clamp to +/- std.
    """
    return std * np.tanh((delta / (std + 1e-8)) * softness)

# def generate_rhythm_sample(mean, std, softness=1.0):
#     """
#     Generates a sample near the mean, softly clamped to +/- std using tanh.
#     """
#     delta = np.random.randn()  # Standard normal
#     return mean + soft_clip_scalar(delta, std, softness)

def generate_rhythm_sample(mean, std):
    delta = np.random.triangular(-1.0, 0.0, 1.0)  # naturally clamped
    return mean + std * delta



def load_stats_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return {
        'DeltaPosition': (df["DeltaPosition"].mean(), df["DeltaPosition"].std()),
        'DurationRatio': (df["DurationRatio"].mean(), df["DurationRatio"].std()),
        'VelocityDelta': (df["VelocityDelta"].mean(), df["VelocityDelta"].std()),
    }

def generate_rhythm_deviations(csv_path, count=100):
    stats = load_stats_from_csv(csv_path)
    samples = []

    for _ in range(count):
        sample = {
            "DeltaPosition": generate_rhythm_sample(*stats["DeltaPosition"]),
            "DurationRatio": generate_rhythm_sample(*stats["DurationRatio"]),
            "VelocityDelta": generate_rhythm_sample(*stats["VelocityDelta"]),
        }
        samples.append(sample)

    return pd.DataFrame(samples)

def plot_distribution_with_stats(df, column, expected_value, xlabel, title):
    mean = df[column].mean()
    std = df[column].std()

    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(expected_value, color='red', linestyle='dashed', linewidth=1.5, label='Expected')
    plt.axvline(mean, color='blue', linestyle='solid', linewidth=2, label=f'Mean: {mean:.3f}')
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=1.5, label=f'+1 Std: {mean + std:.3f}')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=1.5, label=f'-1 Std: {mean - std:.3f}')
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_generated_distributions(df):
    plot_distribution_with_stats(df, "DeltaPosition", 0.0,
                                 "Delta Position (Beats)",
                                 "Generated: Timing Deviation from Expected (0)")
    plot_distribution_with_stats(df, "DurationRatio", 1.0,
                                 "Duration Ratio",
                                 "Generated: Duration Ratio Distribution (Expected 1.0)")
    plot_distribution_with_stats(df, "VelocityDelta", 0.0,
                                 "Velocity Delta",
                                 "Generated: Velocity Delta Distribution (Expected 0)")

if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, "source_data")
    input_csv = os.path.join(data_dir, "rhythm_deviation.csv")
    output_csv = os.path.join(this_dir, "generated_rhythm_deviation.csv")

    df_generated = generate_rhythm_deviations(input_csv, count=200)
    df_generated.to_csv(output_csv, index=False)

    plot_generated_distributions(df_generated)

