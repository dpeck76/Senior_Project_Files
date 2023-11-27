# %% # choose the y-variable
import pandas as pd
y_var = "SR"
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]
# diagnosis_counts = dat_df[possible_diagnoses["Acronym Name"]].sum(axis = 0)
# diagnosis_counts.sort_values()[-10:]
# 1AVB, 2AVB, 2AVB1, 2AVB2, 3AVB, ABI, ALS, APB, AQW, ARS
# AVB, CCR, CR, ERV, FQRS, IDC, IVB, JEB, JPT, LBBB, LBBBB
# LFBBB, LVH, LVQRSAL, LVQRSCL, LVQRSLL, MI, MIBW, MIFW
# MILW, MISW, PRIE, PWC, QTIE, RAH, RBBB, RVH, STDD, STE
# STTC, STTU, TWC, TWO, UW, VB, VEB, VFW, VPB, VPE, VET
# WAVN, WPW, SB, SR, AFIB, ST, AF, SA, SVT, AT, AVNRT, AVRT
# SAAWR

# %% # 1:Import needed libraries
# import plotly.express as px
import os
import numpy as np
# from scipy import signal
import imblearn as il
from tensorflow.keras import backend as K
##### From CSE 450 NN code:
# https://colab.research.google.com/github/byui-cse/cse450-course/blob/master/notebooks/hint_nn.ipynb#scrollTo=KlZiVE696408
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import cv2
# import IPython
# from six.moves import urllib
# print(tf.__version__)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
# %% # 2:Read in the data
file_path = "joined_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","5000","10000","15000","20000","25000","30000","35000","40000","45000","50000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            dat_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            dat_df = pd.concat([dat_df, pd.read_parquet(f"{file_path}{i}.parquet")])

# %% # 3:Create Feature and target dataframes
X = dat_df\
    .drop(
        columns = possible_diagnoses["Acronym Name"]
        )\
    .drop(columns = ["Diagnoses","Diagnoses_list","diagnosis_count", "diagnosis_rowwise_sum", "Record"])\
    .fillna(0)
y = dat_df[possible_diagnoses["Acronym Name"]]
y = y[y_var]

# %% 3b: Select desired features only (instead of using all of them)
# X = X[["median_bpm", "Age", "Sex_M"]]


# %% # 4:Create Train/Test Split
my_seed = 40
X_train, X_test, y_train, y_test = \
    train_test_split(
        X, 
        y, 
        test_size = 0.2, 
        random_state = my_seed)
# %% # 4b: Oversample the data for positive diagnoses
ros = il.over_sampling.RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# %% # Perform min-max Scaling
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)
# transform training data
X_train = norm.transform(X_train)
# transform testing dataabs
X_test = norm.transform(X_test)
# %% # Create Random Forest Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
model = RandomForestClassifier(n_estimators=150)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# %% Print Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
print(f" Accuracy:  {accuracy:.7f}\n",
      f"Precision: {precision:.7f}\n",
      f"Recall:    {recall:.7f}\n",
      f"F1:        {F1:.7f}",)
# %% # Create Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred = np.round(y_pred_probs,0)
# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# %% # Feature Importance Chart
n = 20
feature_importances = model.feature_importances_

# Get the names of features (assuming X contains your feature data)
feature_names = X.columns

# Sort the features by importance
sorted_idx = feature_importances.argsort()

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(range(n), feature_importances[sorted_idx[-n:]])
plt.yticks(range(n), feature_names[sorted_idx[-n:]])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()

# %% # Appendix:Visualization and other code for reference
# Everything in this chunk should be commented out
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

#####
# for i in pos_d:
#     print(i, ": ", dat_df[i].sum(), sep = "", end = "\t")