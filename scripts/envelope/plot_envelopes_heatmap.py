import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# # Load all envelope CSVs (assuming they're named envelope1.csv, envelope2.csv, etc.)
# file_list = sorted(glob.glob("envelope*.csv"))
# envelopes = []

# for file in file_list:
#     df = pd.read_csv(file)
#     envelopes.append(df["Expression"].values)

# # Convert to 2D array
# envelope_matrix = np.array(envelopes)

# plt.imshow(envelope_matrix, aspect='auto', cmap='viridis', interpolation='none')
# plt.colorbar(label="Expression")
# plt.xlabel("Position Index")
# plt.ylabel("Envelope Index")
# plt.title("Heatmap of 32 MIDI Expression Envelopes")
# plt.show()

# Parameters
target_length = 100

# Load and resample
file_list = sorted(glob.glob("envelope*.csv"))
envelopes = []

for file in file_list:
    df = pd.read_csv(file)

    # Original positions and values
    orig_x = df["Position"].values
    orig_y = df["Expression"].values

    # Create evenly spaced positions in the same range
    new_x = np.linspace(orig_x.min(), orig_x.max(), target_length)

    # Native numpy interpolation
    new_y = np.interp(new_x, orig_x, orig_y)

    envelopes.append(new_y)

# Convert to 2D NumPy array
envelope_matrix = np.array(envelopes)

# Plot heatmap
plt.imshow(envelope_matrix, aspect='auto', cmap='viridis', interpolation='none')
plt.colorbar(label="Expression")
plt.xlabel("Normalized Position Index")
plt.ylabel("Envelope Index")
plt.title("Heatmap of 32 MIDI Expression Envelopes (Resampled)")
plt.show()
