import os
import glob
import pandas as pd

#data_dir = "./viz_batch"
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "viz_batch")

files = sorted(glob.glob(os.path.join(data_dir, "generated_morph2_*.csv")))
# files = sorted(glob.glob(os.path.join(data_dir, "generated_*.csv")))


print(f"Found {len(files)} CSV files.")

for file in files:
    print(f"\n File: {file}")
    try:
        df = pd.read_csv(file)
        print(f" Columns: {df.columns.tolist()}")
        print(f" Length: {len(df)}")
        print(f" First few rows:\n{df.head()}")
        print(f" Last row:\n{df.tail(1)}")
    except Exception as e:
        print(f"Error: Failed to read {file}: {e}")
