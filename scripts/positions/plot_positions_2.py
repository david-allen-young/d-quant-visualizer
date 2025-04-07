import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_position_offset_histogram(csv_path, bins=30):
    """
    Histogram of DeltaPosition (deviation from NominalPosition).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["DeltaPosition"], bins=bins, kde=True, color='blue', alpha=0.6)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label="Expected Position")
    plt.xlabel("Delta Position (Beats)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Timing Deviations")
    plt.legend()
    plt.grid()
    plt.show()

def plot_offset_boxplot_by_position(csv_path):
    """
    Boxplot of DeltaPosition grouped by NominalPosition (rounded).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["RoundedPosition"] = df["NominalPosition"].round()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="RoundedPosition", y="DeltaPosition", data=df)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title("Timing Deviation by Nominal Position")
    plt.xlabel("Nominal Beat (Rounded)")
    plt.ylabel("Delta Position (Beats)")
    plt.grid()
    plt.show()

def plot_ratio_and_velocity_trends(csv_path):
    """
    Line plots for DurationRatio and VelocityDelta over time.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    sns.lineplot(ax=axs[0], x="NominalPosition", y="DurationRatio", data=df, marker="o")
    axs[0].set_title("Duration Ratio over Time")
    axs[0].set_ylabel("Duration Ratio")
    axs[0].grid()

    sns.lineplot(ax=axs[1], x="NominalPosition", y="VelocityDelta", data=df, marker="o", color="purple")
    axs[1].set_title("Velocity Delta over Time")
    axs[1].set_xlabel("Nominal Position (Beats)")
    axs[1].set_ylabel("Velocity Delta")
    axs[1].grid()

    plt.tight_layout()
    plt.show()

# Example usage:
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "source_data")
csv_path = os.path.join(data_dir, "rhythm_deviation.csv")

plot_position_offset_histogram(csv_path)
plot_offset_boxplot_by_position(csv_path)
plot_ratio_and_velocity_trends(csv_path)
