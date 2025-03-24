import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_envelope(csv_file):
    df = pd.read_csv(csv_file)
    plt.plot(df['Position'], df['Expression'])
    plt.title("Breath Controller Envelope")
    plt.xlabel("Time (beats)")
    plt.ylabel("Controller Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(this_dir, "../../sample_data/envelope_example.csv"))
    print(f"Reading file from: {csv_path}")
    plot_envelope(csv_path)
