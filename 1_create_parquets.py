# %% # 1:Import needed Libraries
import numpy as np
import pandas as pd
import wfdb
import os
import scipy.io as io
from random import shuffle
from random import seed
# %% # 2:Create the "all_files" list
# Replace with your own file path to this file location
os.chdir("C:/users/David/Documents/David BYU-Idaho/Fall 2023/DS 499/Senior_Project_Files")
folder_path = "..\\electrocardiogram-database-arrhythmia-study"
all_files = []
for root, directories, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        all_files.append(file_path)
all_files = all_files\
    .remove("..\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\01\019\JS01052.mat")\
    .remove("..\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\42\424\JS41800.mat")
# %% # 3:Pull the .hea files and the .mat files out of "all_files"
# For each of the files in all_files, read in the ones that end in ".mat". 
mat_files = [i for i in all_files if i.endswith(".mat")]
# Randomize the order
seed(40)
shuffle(mat_files) 
# For the .hea files, take off the ".hea" because that is how the function receives it. 
hea_files = [i[:-4] for i in mat_files]
# %% # 4:Determine how many of the patients' readings to use
num_rows = 45152 # Use all of the data (num_rows = 45,152) to provide the best perfomance of the model; use less if testing or otherwise want a quicker result than you would get if you use it all
initial_chunk_size = 4520 # Change this to whatever size you want/believe your computer can handle. I chose 4,520 because it was a good size for my computer's RAM. 
hea_files = hea_files[:num_rows] # Change the files we are using to the num_rows variable value
mat_files = mat_files[:num_rows]
# %% # 5:Read all of the data and save them into a list object; first with records - 
# How many records to read in at a time
chunk_size = initial_chunk_size
start_row = 0
while start_row < num_rows:
    chunk_size = chunk_size if start_row + chunk_size < num_rows else num_rows - start_row
    records = []
    for i in range(start_row, start_row + chunk_size):
        try:
            records.append(wfdb.rdrecord(hea_files[i]))
        except:
            print(hea_files[i])
    ### Text Wrangling
    # Make the list into a pandas dataframe
    records_df = pd.DataFrame(records)\
        .rename(columns={0: "Record"})
    # Create empty list variables to later make into columns
    age, sex, diagnoses, record = [[],[],[],[]]
    for i in range(len(records)):
        sex.append(records[i].comments[1][5:])
        try:
            age.append(int(records[i].comments[0].strip("Age: ")))
        except:
            age.append(np.nan)
        diagnoses.append(records[i].comments[2].strip("Dx: "))
        record.append(mat_files[start_row + i][-11:-4])
    # Add the lists to the dataframe as columns
    records_df = records_df\
        .assign(Diagnoses = diagnoses)\
        .assign(Diagnoses_list = list(map(lambda x: x.split(sep = ","), diagnoses)))\
        .assign(Age = age)\
        .assign(Sex = sex)\
        .assign(Record = record)
    ### Create a column for each possible diagnosis in the database with 1 or 0 values for each of the rows
    possible_diagnoses = pd.read_csv(r"..\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
    # this is a df with 3 columns:
    # 1: The Snomed numeric code for the condition
    # 2: The Snomed acronym
    # 3: The full condition name
    for i in range(possible_diagnoses.shape[0]):
        records_df[possible_diagnoses["Acronym Name"][i]]\
         = records_df["Diagnoses_list"]\
            .apply(lambda x: int(str(possible_diagnoses["Snomed_CT"][i])in x))
    ### Make 2 diagnosis count columns: 
    # One from the list of diagnoses; the other from the new diagnosis columns
    # This checks to see if there are diagnoses that are not in the list provided with the database - it turns out there are.
    records_df = records_df\
        .assign(diagnosis_rowwise_sum = records_df\
            .loc[:,possible_diagnoses["Acronym Name"]]\
                .sum(axis = 1))\
        .assign(diagnosis_count = \
            records_df['Diagnoses_list']\
                .apply(lambda x: len(x)))
    # Write the records .parquet file
    records_df.to_parquet(f"records_{start_row}.parquet")
    start_row += chunk_size

# %% # Put the readings into .parquet files
chunk_size = initial_chunk_size
start_row = 0
while start_row < num_rows:
    chunk_size = chunk_size if start_row + chunk_size < num_rows else num_rows - start_row
    readings, record = ([],[])
    for i in range(start_row, start_row + chunk_size):
        # Print out a message each time we are 1 additional % done with the entire data
        if i % (451) == 0:
            print(f"{i // 451}% done")
        try:
            readings.append(io.loadmat(mat_files[i]))
        except:
            print(mat_files[i])
        record.append(mat_files[i][-11:-4])
    # Make it into a dataframe
    readings_df = pd.DataFrame(readings)
    readings_df = readings_df\
        .assign(Record = record)
    readings_df = readings_df\
        .drop(readings_df.index[readings_df["Record"] == "JS01052"])
    size = readings_df.shape[0]
    readings_df = readings_df\
        .explode("val")\
        .assign(Type = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]*(size))

    readings_df.to_parquet(f'readings_{start_row}.parquet', index=False)    
    print(f'Parquet file readings_{start_row}.parquet created successfully.')
    start_row += chunk_size