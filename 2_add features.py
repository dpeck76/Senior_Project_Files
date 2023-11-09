# %% # 1:Import needed libraries
import numpy as np
import pandas as pd
import wfdb
import os
import scipy.io as io
from scipy.signal import find_peaks
# %% # 2:Read in records_df (created in "1_create_parquet_files.py")
records_df = pd.read_parquet("records.parquet")

# %% # 3:Read in readings_df (created in "1_create_parquet_files.py" as several parquet files)
file_path = "readings_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            readings_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            readings_df = pd.concat([readings_df, pd.read_parquet(f"{file_path}{i}.parquet")])
# %% # 4:Make various summary statistics of the different readings
dat = readings_df["val"]
readings_df["mean"] = dat.apply(np.mean)
readings_df["median"] = dat.apply(np.median)
readings_df["std"] = dat.apply(np.std)
readings_df["max"] = dat.apply(np.max)
readings_df["min"] = dat.apply(np.min)
readings_df["len"] = dat.apply(len)
# %% # 5:Make summary statistics using the find_peaks function
readings_df["num_beats"] = dat.apply(
  lambda x: 
  len(
    find_peaks(
      x, 
      prominence=(
        np.max(x) - np.min(x)
      ) * 0.38
    )[0]
  )                                 )
readings_df["bpm"] = \
  readings_df["num_beats"] * \
  30000 / \
  readings_df["len"]
# %% # 6:Define a list variable with all of the feature column names
feature_cols = ["mean", "median", "std", 
            "max", "min", "len",
            "num_beats", "bpm"]
# %% # 7:Pivot the table
readings_df = readings_df\
  .pivot(
    index = "Record", 
    columns = "Type", 
    values = feature_cols)
# %% # 8:Rename columns of the pivoted dataframe so that they are menaingful again
# All of the different types of readings (corresponding to different places on the body)
type = ["I",  "II","III","aVR", 
        "aVL","aVF","V1", "V2", 
        "V3", "V4", "V5", "V6"]
# make an empty list that will contain the column names for the pivoted table
rdf_columns = []
for i in feature_cols:
    for j in type:
        rdf_columns.append(j + "_" + i)
readings_df.columns = rdf_columns
readings_df = readings_df.reset_index()
# %% # 9:Join the 2 dataframes
# Convert the "Record" columns to strings
readings_df["Record"] = readings_df["Record"].astype(str)
records_df["Record"] = records_df["Record"].astype(str)
# Perform an inner join
result_df = readings_df.merge(records_df, on="Record", how="inner")

# %% # 10:Create Several parquet files so that 1: if it crashes partway through, we don't lose all of the work and 2: the files are a more manageable size
# n is the number of lines you want to write at a time
n = 50000  # Adjust this to your desired chunk size
output_path = 'result_test'
total_rows = len(result_df)
start_row = 0
while start_row < total_rows:
    chunk = result_df[start_row:start_row + n]
    chunk.to_parquet(f'{output_path}_{start_row}.parquet', index=False)    
    print(f'Parquet file {output_path}_{start_row}.parquet created successfully.')
    start_row += n
# readings_df.to_parquet(f"{file_path}0.parquet")

# %% # Appendix:Additional code for reference
# pd.read_parquet(f"{file_path}0.parquet").head()

# %% 