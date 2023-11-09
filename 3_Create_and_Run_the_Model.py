# %% # Import needed libraries
import plotly.express as px
import os
import numpy as np
import pandas as pd
from scipy import signal
# %% # Read in the data
file_path = "result_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            dat_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            dat_df = pd.concat([dat_df, pd.read_parquet(f"{file_path}{i}.parquet")])


# %% # example visualization code
# dat = dat_df["val"][1+12*29]
# peak_indices, _ \
#     = signal.find_peaks(
#         dat, 
#         prominence = (np.max(dat)-np.min(dat))*0.38)

# px.scatter(
#     x = peak_indices, 
#     y = dat[peak_indices],
#     color_discrete_sequence=['red'] 
# ).add_trace(
#     px.line(
#         x = range(len(dat)),
#         y = dat
#     ).data[0]
# )


# %% # Create the Machine Learning model

