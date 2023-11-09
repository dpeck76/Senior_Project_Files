# %% # 1:Import needed Libraries
import numpy as np
import pandas as pd
import wfdb
import os
import scipy.io as io
import pyarrow
# signal processing
# %% # 2:Create the "all_files" list
folder_path = "../electrocardiogram-database-arrhythmia-study"
all_files = []
for root, directories, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        all_files.append(file_path)
# %% # 3:Pull the .hea files and the .mat files out of "all_files"
# For each of the files in all_files, read in the ones that 
# end in ".hea". Also, take off the ".hea" because that is 
# how the function receives it. 
hea_files = [i[:-4] for i in all_files if i.endswith(".hea")]
# Do the same thing for the ".mat" files. 
mat_files = [i for i in all_files if i.endswith(".mat")]
# %% # 4:Determine how many of the patients' readings to use
# We anticipate using all of this by the end to give the 
# model all of the available data. However, for right now, 
# we will only use a small part of it for testing purposes
num_rows = 50
hea_files = hea_files[:num_rows]
mat_files = mat_files[:num_rows]
# %% # 5:Read all of the data and save them into a list object; first with reacords - 
# This usually takes around 25 minutes to complete with the whole dataset
records = []
print("starting hea files")
for i in range(len(hea_files)):
    if i in range(0, len(hea_files), 451):
        print(f"{i}: {round((1-((45152-i)/45152))*100,0)}% done")
    try:
        records.append(wfdb.rdrecord(hea_files[i]))
    except:
        print(hea_files[i])
# %% # 6:Do some text wrangling to get needed data
records_df = pd.DataFrame(records)\
    .rename(columns={0: "Record"})
age, sex, diagnoses, record = [[],[],[],[]]
for i in range(len(records)):
    sex.append(records[i].comments[1][5:])
    try:
        age.append(int(records[i].comments[0].strip("Age: ")))
    except:
        age.append(np.nan)
    diagnoses.append(records[i].comments[2].strip("Dx: "))
    record.append(mat_files[i][-11:-4])
# %% # 7:Make some extra columns that we will likely have need of later
records_df = records_df\
    .assign(Diagnoses = diagnoses)\
    .assign(Diagnoses_list = list(map(lambda x: x.split(sep = ","), diagnoses)))\
    .assign(Age = age)\
    .assign(Sex = sex)\
    .assign(Record = record)
# %% # 8:Create a column for each possible diagnosis in the database with 1 or 0 values for each of the rows
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
for i in range(possible_diagnoses.shape[0]):
    my_list = []
    for j in range(records_df.shape[0]):
        if str(possible_diagnoses["Snomed_CT"][i]) in records_df["Diagnoses_list"][j]:
            my_list.append(1)
        else:
            my_list.append(0)
    records_df[possible_diagnoses["Acronym Name"][i]] = my_list
# %% # 9:Make 2 diagnosis count columns: 
# One from the list of diagnoses; the other from the new diagnosis columns
# This checks to see if there are diagnoses that are not in the list provided with the database - it turns out there are.
records_df = records_df\
    .assign(diagnosis_rowwise_sum = records_df.drop(["Diagnoses","Diagnoses_list","Age","Sex","Record"], axis = 1).sum(axis = 1))\
    .assign(diagnosis_count = records_df['Diagnoses_list'].apply(lambda x: len(x)))
# %% # 10:Write the records .parquet file
records_df.to_parquet("records.parquet")
# %% # 11:Now with readings
readings = []
print("starting mat files")
for i in range(len(mat_files)):
    if i in range(0, len(hea_files), 451):
        print(f"{round((1-(45152-i)/(45152))*100,0)}% done")
    try:
        readings.append(io.loadmat(mat_files[i]))
    except:
        print(mat_files[i])

# %% # 12:Make readings into a Pandas df and create needed columns
# .drop(index = [999])\
readings_df = pd.DataFrame(readings)\
    .assign(Record = record)\
    .explode("val")\
    .assign(Type = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]*(len(readings)))
# %% # 13:Write readings .parquet file(s)
# n is the number of lines you want to write at a time
n = 50000  # Adjust this to your desired chunk size
output_path = 'readings_test'

total_rows = len(readings_df) # 
start_row = 0 # This will interate with each time throuh the while loop

while start_row < total_rows:
    chunk = readings_df[start_row:start_row + n]
    chunk.to_parquet(f'{output_path}_{start_row}.parquet', index=False)    
    print(f'Parquet file {output_path}_{start_row}.parquet created successfully.')
    start_row += n




#readings_df[:500000].to_parquet("readings.parquet")

# %% # Appendix: Commented code for reference
# for i in range(len(mat_files)):
#     if mat_files[i] == "./electrocardiogram-database-arrhythmia-study\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\WFDBRecords\\01\\019\\JS01052.mat":
#         print(i)
#     if mat_files[i] == "./electrocardiogram-database-arrhythmia-study\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\WFDBRecords\\23\\236\\JS23074.mat":
#         print(i)
# %%