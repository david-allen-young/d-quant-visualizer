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
    plot_envelope("../d-quant/assets/output_csv/envelope1.csv")
