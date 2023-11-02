# %% 
import numpy as np
import pandas as pd
import wfdb
import os
import scipy.io as io
from scipy.signal import find_peaks
# %% # Read in records_df
records_df = pd.read_parquet("records.parquet")

# %% read in readings_df
file_path = "readings_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            readings_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            readings_df = pd.concat([readings_df, pd.read_parquet(f"{file_path}{i}.parquet")])
# %%
# All of the different types of readings (corresponding to different places on the body)
type = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Make various summary statistics of the different readings
readings_df["mean"] = readings_df["val"].apply(np.mean)
readings_df["median"] = readings_df["val"].apply(np.median)
readings_df["std"] = readings_df["val"].apply(np.std)
readings_df["max"] = readings_df["val"].apply(np.max)
readings_df["min"] = readings_df["val"].apply(np.min)
readings_df["len"] = readings_df["val"].apply(len)
# readings_df.len.value_counts() # - Double checks that we have the same number of readings for each observation

# %%
# n is the number of lines you want to write at a time
n = 50000  # Adjust this to your desired chunk size
output_path = 'readings_test'
total_rows = len(readings_df)
start_row = 0
while start_row < total_rows:
    chunk = readings_df[start_row:start_row + n]
    chunk.to_parquet(f'{output_path}_{start_row}.parquet', index=False)    
    print(f'Parquet file {output_path}_{start_row}.parquet created successfully.')
    start_row += n
# readings_df.to_parquet(f"{file_path}0.parquet")

# %%
pd.read_parquet(f"{file_path}0.parquet").head()

# %%

