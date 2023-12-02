# %% # 1:Import needed libraries
import numpy as np
import pandas as pd
# import wfdb
import os
from scipy.signal import find_peaks
# %% # 2:Read in records_df (created in "1_create_parquet_files.py")
feature_cols = ["mean", "median", "std",
              "max", "min", "len", "bpm",
              "std_elapsed_time", "median_elapsed_time"]
# All of the different types of readings (corresponding to different places on the body)
type = ["I",  "II","III","aVR", 
        "aVL","aVF","V1", "V2", 
        "V3", "V4", "V5", "V6"]
# make an empty list that will contain the column names for the pivoted table
rdf_columns = []
for i in feature_cols:
    for j in type:
        rdf_columns.append(j + "_" + i)
# %% # 3:Read in readings_df (created in "1_create_parquet_files.py" as several parquet files)
os.chdir("C:/users/David/Documents/David BYU-Idaho/Fall 2023/DS 499/Senior_Project_Files")
readings_file_paths = [file for file in os.listdir(os.getcwd()) if file.startswith("readings") and file.endswith(".parquet")]
records_file_paths = [file for file in os.listdir(os.getcwd()) if file.startswith("records") and file.endswith(".parquet")]
readings_file_paths.remove('readings_5000.parquet')
# %%
for reading, record  in zip(readings_file_paths, records_file_paths):
  try:
    # print(i)
    # if os.path.exists(f"{file_path}{i}.parquet"):
    readings_df = pd.read_parquet(reading)
    records_df = pd.read_parquet(record)
      
    # Turn Sex column into 1s and 0s
    records_df["Sex_M"] = \
      records_df["Sex"]\
        .apply(
          lambda x: 1 if x == "Male" else 0
        )
    records_df = records_df.drop(["Sex"], axis = 1)
    # 4:Make various summary statistics of the different readings
    dat = readings_df["val"]
    readings_df["mean"] = dat.apply(np.mean)
    readings_df["median"] = dat.apply(np.median)
    readings_df["std"] = dat.apply(np.std)
    readings_df["max"] = dat.apply(np.max)
    readings_df["min"] = dat.apply(np.min)
    readings_df["len"] = dat.apply(len)
    # 5:Make summary statistics using the find_peaks function
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
    readings_df = readings_df.drop("num_beats", axis = 1)
    readings_df["peak_indices"] = \
      dat.apply(
      lambda x:
        find_peaks(
          x,
          prominence=(
            np.max(x) - np.min(x))*0.38
        )[0]
      )
    readings_df["diffs"] = \
      readings_df["peak_indices"]\
      .apply(
        lambda x:[x[i] - x[i-1]for i in range(1, len(x))]
      )
    readings_df["median_elapsed_time"] = \
      readings_df["diffs"].apply(
        np.median  
      ).fillna(0)
    readings_df["std_elapsed_time"] = \
      readings_df["diffs"].apply(
        np.std  
      ).fillna(0)
    readings_df = readings_df.drop(["diffs", "peak_indices"], axis = 1)
    # 6:Define a list variable with all of the feature column names
    # feature_cols = ["mean", "median", "std",
    #             "max", "min", "len", "bpm",
    #             "std_elapsed_time", "median_elapsed_time"]
    # 7:Pivot the table
    # this will make is so that there is only one row per patient instead of 1 row per reading
    readings_df_pivoted = readings_df\
      .pivot(
        index = "Record", 
        columns = "Type", 
        values = feature_cols)
    # 8:Rename columns of the pivoted dataframe so that they are menaingful again
    readings_df_pivoted.columns = rdf_columns
    readings_df = readings_df_pivoted.reset_index()

    # 8b:Create Median BPM column
    readings_df["median_bpm"]\
      = readings_df[[i + "_bpm" for i in type]]\
        .median(axis=1)

    # 9:Join the 2 dataframes
    # Convert the "Record" columns to strings
    readings_df["Record"] = readings_df["Record"].astype(str)
    records_df["Record"] = records_df["Record"].astype(str)
    # Perform an inner join
    joined_df = readings_df.merge(records_df, on="Record", how="inner")
    # Print out the shape of the data frames to ensure that the joins are going as expected
    print(f"readings_df from {reading} shape:", readings_df.shape, end = "")
    print(f"\nrecords_df from {record} shape:", records_df.shape, end = "")
    print(f"\njoined_df shape:", joined_df.shape)
    # 10:Create Several parquet files so that 1: if it crashes partway through, we don't lose all of the work and 2: the files are a more manageable size
    # n is the number of lines you want to write at a time
    joined_df.to_parquet(f'joined_{reading[9:]}', index=False)    
    print(f'Parquet file joined_{reading[9:]} created successfully.')
  except:
     print(f"Either {record} or {reading} isn't reading in right")


# readings_df.to_parquet(f"{file_path}0.parquet")

# %% # Appendix:Additional code for reference
# pd.read_parquet(f"{file_path}0.parquet").head()

# %% 