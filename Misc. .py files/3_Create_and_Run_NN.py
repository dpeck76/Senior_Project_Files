# %% # choose the y-variable
import pandas as pd
y_var = "AFIB"
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
# %% # 2:Read in the data
file_path = "result_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
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
    .drop(columns = ["Diagnoses","Diagnoses_list","Record"])\
    .fillna(0)

y = dat_df[possible_diagnoses["Acronym Name"]]
y = y[y_var]

# %% # 4:Create Train/Test Split
my_seed = 40
X_test, X_train, y_test, y_train = \
    train_test_split(
        X, 
        y, 
        test_size = 0.8, 
        random_state = my_seed)
# %% # 4b: Oversample the data for positive diagnoses
ros = il.over_sampling.RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# %% # Perform min-max Scaling
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)
# transform training data
X_train = norm.transform(X_train)
# transform testing dataabs
X_test = norm.transform(X_test)
# %% # Create a sequential model NN

# Define the model
model = Sequential()

# Input layer
model.add(Dense(100, input_shape=(X_train.shape[1],), activation='relu'))

# Hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dense(42, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# model = Sequential()
# model.add(Dense(128, input_dim=len(X_train[0]), activation='leaky_relu'))
# # model.add(Dropout(.5))
# model.add(Dense(192, activation='relu'))
# model.add(Dense(256, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(1, activation='relu'))

# %% #

# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
# %% # create an optimizer and the additional metrics
opt = keras.optimizers.Adam(learning_rate=0.001)
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return  true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return  true_positives / (possible_positives + K.epsilon())

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())
# %%

# Compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200)
history = \
    model.fit(X_train, 
              y_train, 
              epochs=100, 
              validation_split=.35, 
              batch_size=32, 
              callbacks=[early_stop],
              shuffle=False)
# %% # Visualize
metric = "precision" # set to either "accuracy" or "loss" depending on the visualization you want
import plotly.express as px
loss_values = history.history[metric]
val_loss_values = history.history["val_" + metric]
metric = metric.capitalize()
# Plot the loss over time
plt.plot(loss_values, label='Training ' + metric)
plt.plot(val_loss_values, label='Validation ' + metric)
plt.title(metric + ' Over Time')
plt.xlabel('Epochs')
plt.ylabel(metric)
plt.legend()
plt.show()

# %% # Defining y_pred variables
y_pred_probs = model.predict(
    [X_test])
# y_pred_rounded = np.round(y_pred_probs)
y_pred_probs[y_pred_probs > 0.5] = 1
y_pred_probs[y_pred_probs <= 0.5] = 0
y_pred = np.round(y_pred_probs)

# %% # Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
print(f" Accuracy:  {accuracy:.7f}\n",
      f"Precision: {precision:.7f}\n",
      f"Recall:    {recall:.7f}\n",
      f"F1:        {F1:.7f}",)

# %% # Create a confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
import seaborn as sns
# Display the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Confusion Matrix for {y_var}")
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