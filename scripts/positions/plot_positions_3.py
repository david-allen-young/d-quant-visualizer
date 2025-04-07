import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_distribution_with_stats(df, column, expected_value, xlabel, title):
    """
    Generic helper for plotting a distribution with mean and +/-1 std lines.
    """
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
    plt.show()

def plot_all_distributions(csv_path):
    """
    Load rhythm deviation CSV and plot histograms with mean +/-1 stddev for:
    - DeltaPosition (timing deviation)
    - DurationRatio (note length deviation)
    - VelocityDelta (dynamics deviation)
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plot_distribution_with_stats(df, "DeltaPosition", 0.0,
                                 "Delta Position (Beats)",
                                 "Timing Deviation from Expected (0)")

    plot_distribution_with_stats(df, "DurationRatio", 1.0,
                                 "Duration Ratio",
                                 "Duration Ratio Distribution (Expected 1.0)")

    plot_distribution_with_stats(df, "VelocityDelta", 0.0,
                                 "Velocity Delta",
                                 "Velocity Delta Distribution (Expected 0)")

if __name__ == "__main__":
    # Adjust these paths as needed
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, "source_data")
    csv_path = os.path.join(data_dir, "rhythm_deviation.csv")

    plot_all_distributions(csv_path)
