import os
import pandas as pd
import matplotlib.pyplot as plt

# Load data
this_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.abspath(os.path.join(this_dir, "../../../d-quant/assets/output_csv/envelope17.csv"))
print(f"Reading file from: {csv_path}")
df = pd.read_csv(csv_path)

# Plot
plt.plot(df["Position"], df["Expression"], marker='o', linestyle='-')
plt.xlabel("Position")
plt.ylabel("Expression (0-1)")
plt.title("MIDI Expression Envelope")
plt.show()

