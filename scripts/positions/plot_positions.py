import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_position_deviation_histogram(position_deltas, bins=30):
    """
    Plots a histogram of position deviations using Seaborn and Matplotlib.
    
    :param position_deltas: List or NumPy array of timing deviations (in beats).
    :param bins: Number of bins for the histogram.
    """
    # Create a DataFrame for Seaborn compatibility
    df = pd.DataFrame({"Position Delta": position_deltas})

    # Plot histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Position Delta"], bins=bins, kde=True, color='blue', alpha=0.6)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label="Expected Timing")
    plt.xlabel("Position Delta (Beats)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Position Deviations")
    plt.legend()
    plt.grid()
    plt.show()

# Example Usage:
# Generate synthetic timing deviations (normally distributed)
num_samples = 500
synthetic_position_deltas = np.random.normal(loc=0, scale=0.1, size=num_samples)

# Call the function with sample data
plot_position_deviation_histogram(synthetic_position_deltas)
