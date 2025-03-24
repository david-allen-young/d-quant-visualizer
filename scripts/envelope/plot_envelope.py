import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("envelope17.csv")

# Plot
plt.plot(df["Position"], df["Expression"], marker='o', linestyle='-')
plt.xlabel("Position")
plt.ylabel("Expression (0-1)")
plt.title("MIDI Expression Envelope")
plt.show()
