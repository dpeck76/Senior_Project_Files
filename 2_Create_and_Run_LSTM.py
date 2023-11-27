# %% # 1:Import needed libraries
# import plotly.express as px
import os
import numpy as np
import pandas as pd
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
# %% # choose the y-variable
y_var = "SB"
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]

# 1AVB, 2AVB, 2AVB1, 2AVB2, 3AVB, ABI, ALS, APB, AQW, ARS
# AVB, CCR, CR, ERV, FQRS, IDC, IVB, JEB, JPT, LBBB, LBBBB
# LFBBB, LVH, LVQRSAL, LVQRSCL, LVQRSLL, MI, MIBW, MIFW
# MILW, MISW, PRIE, PWC, QTIE, RAH, RBBB, RVH, STDD, STE
# STTC, STTU, TWC, TWO, UW, VB, VEB, VFW, VPB, VPE, VET
# WAVN, WPW, SB, SR, AFIB, ST, AF, SA, SVT, AT, AVNRT, AVRT
# SAAWR


# %% # 2:Read in the data
file_path = "readings_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            dat_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            dat_df = pd.concat([dat_df, pd.read_parquet(f"{file_path}{i}.parquet")])

file_path = "result_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            y_dat_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            y_dat_df = pd.concat([dat_df, pd.read_parquet(f"{file_path}{i}.parquet")])
# %% # 3:Create Feature and target dataframes
type = ["I",  "II","III","aVR", 
        "aVL","aVF","V1", "V2", 
        "V3", "V4", "V5", "V6"]
X = dat_df\
    .pivot(
        index = "Record", 
        columns = "Type", 
        values = "val")
y = y_dat_df[possible_diagnoses["Acronym Name"]]
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
X_train, y_train = ros.fit_resample(X_train, y_train)


# %% # Perform min-max Scaling
# Convert from a df of arrays to a 3d array
X_train = np.array(X_train.to_numpy().tolist())
X_test = np.array(X_test.to_numpy().tolist())
# Make 2d to be able to standardize the values
X_train_2D = X_train.reshape(-1, X_train.shape[-1])
X_test_2D = X_test.reshape(-1, X_test.shape[-1])
# Apply MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled_2D = scaler.fit_transform(X_train_2D)
X_test_scaled_2D = scaler.fit_transform(X_test_2D)
# Reshape back to 3D
X_train = X_train_scaled_2D.reshape(X_train.shape)
X_test = X_test_scaled_2D.reshape(X_test.shape)

# %% # Create a sequential model LSTM
time_steps = X_train.shape[2]
features = X_train.shape[1]
# Define the model
model = Sequential()
model.add(LSTM(units=50, input_shape=(features, time_steps)))
model.add(Dense(units=1, activation='sigmoid'))


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
# %% # Compile  and train the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])
# This makes it so that if it is not improving for 200 epochs, it will not continue
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200)
# This is the step that takes a long time

checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)


history = \
    model.fit(X_train, 
              y_train, 
              epochs=2000, 
              validation_split=.35, 
              batch_size=32, 
              callbacks=[early_stop, model_checkpoint],
              shuffle=False)
# %% # Visualize metrics over time
metric = "loss" # set to "accuracy", "f1", "precision", "recall", or "loss" depending on the visualization you want
import plotly.express as px
loss_values = history.history[metric]
val_loss_values = history.history["val_" + metric]
metric = metric.capitalize()
# Plot the chosen metric over time
plt.plot(loss_values, label='Training ' + metric)
plt.plot(val_loss_values, label='Validation ' + metric)
plt.title(metric + ' Over Time')
plt.xlabel('Epochs')
plt.ylabel(metric)
plt.legend()
plt.show()

# %% # Confusion Matrix
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

# %%
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
f1 = f1_score(y_true = y_test, y_pred = y_pred)
recall = recall_score(y_true = y_test, y_pred = y_pred)
precision = precision_score(y_true = y_test, y_pred = y_pred)
print("accuracy: ", round(accuracy, 5), "\n",
      "precision:", round(precision, 5), "\n",
      "recall: ", round(recall, 5), "\n",
      "F1:", round(f1, 5),"\n",
      sep = "")

# %% # Appendix: Visualization and other code for reference
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