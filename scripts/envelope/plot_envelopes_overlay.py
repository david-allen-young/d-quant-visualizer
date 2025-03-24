import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Parameters
target_length = 100

# Load and interpolate all envelopes
file_list = sorted(glob.glob("envelope*.csv"))
interpolated_envelopes = []

for file in file_list:
    df = pd.read_csv(file)

    # Original data
    orig_x = df["Position"].values
    orig_y = df["Expression"].values

    # Interpolated x-axis (normalized scale)
    interp_x = np.linspace(orig_x.min(), orig_x.max(), target_length)
    interp_y = np.interp(interp_x, orig_x, orig_y)

    interpolated_envelopes.append(interp_y)

# Convert to 2D array for averaging and std dev
envelope_matrix = np.array(interpolated_envelopes)
mean_envelope = envelope_matrix.mean(axis=0)
std_envelope = envelope_matrix.std(axis=0)

# X-axis for plotting
x_vals = np.linspace(0, 1, target_length)

# Plot all individual envelopes
plt.figure(figsize=(12, 6))
for y_values in envelope_matrix:
    plt.plot(x_vals, y_values, alpha=0.3, linewidth=1)

# Plot shaded standard deviation band
plt.fill_between(x_vals,
                 mean_envelope - std_envelope,
                 mean_envelope + std_envelope,
                 color='gray',
                 alpha=0.3,
                 label='+/-1 Std Dev')

# Plot mean envelope
plt.plot(x_vals, mean_envelope, color='black', linestyle='--', linewidth=2, label='Mean Envelope')

# Labels and legend
plt.xlabel("Normalized Position (0-1)")
plt.ylabel("Expression (0-1)")
plt.title("Overlay of 32 MIDI Expression Envelopes with Mean and Std Dev")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
