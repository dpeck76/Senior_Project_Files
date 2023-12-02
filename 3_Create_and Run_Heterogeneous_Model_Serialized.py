# %% # 1:Import needed libraries
# import plotly.express as px
import seaborn as sns
import os
import numpy as np
import pandas as pd
# from scipy import signal
import imblearn as il
from tensorflow.keras import backend as K
##### From CSE 450 NN code:
# https://colab.research.google.com/github/byui-cse/cse450-course/blob/master/notebooks/hint_nn.ipynb#scrollTo=KlZiVE696408
# TensorFlow and tf.keras
# import tensorflow as tf
from tensorflow import keras
# import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import cv2
# import IPython
# from six.moves import urllib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, LSTM
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
# %% # Choose y-variable
y_var = "SB"
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]

# %% # Read in the parquet files as the X variable
def load_and_preprocess_data(joined_file_path, readings_file_path, y_var = pos_d):
    X, X2, y, joined_dat_df, readings_dat_df = (None,None,None,None,None)
    joined_dat_df = pd.read_parquet(joined_file_path)
    readings_dat_df = pd.read_parquet(readings_file_path)
    X2 = joined_dat_df\
        .drop(pos_d, axis = 1)\
        .drop(["Record", "diagnosis_rowwise_sum", "diagnosis_count", "Diagnoses", "Diagnoses_list"], axis = 1)\
        .fillna(0)
    y = joined_dat_df[y_var]
    X = \
        np.transpose(
            np.array(
                readings_dat_df\
                    #.drop(readings_dat_df.index[readings_dat_df["Record"] == "JS01052"])\
                    .pivot(
                        index = "Record", 
                        columns = "Type", 
                        values = "val")\
                    .fillna(0)\
                    .to_numpy()\
                    .tolist())\
                    .reshape((-1, 12, 5000))
                , (0, 2, 1))
    return X, X2, y



# %%
# columns_to_keep = ["Age", "Sex_M"]
# type = ["I",  "II","III","aVR", 
#         "aVL","aVF","V1", "V2", 
#         "V3", "V4", "V5", "V6"]
# for i in type:
#     columns_to_keep.append(f"{i}_std_elapsed_time")
# for i in type:
#     columns_to_keep.append(f"{i}_median_elapsed_time")
# X2 = result_df[columns_to_keep].fillna(0)


# %% # reandomly choose a file to be the test file
from random import choice
my_seed = 40
joined_files = [file for file in os.listdir(os.getcwd()) if file.startswith("joined_") and file.endswith(".parquet")]
# joined_files.remove("joined_5000.parquet")
readings_files = [file for file in os.listdir(os.getcwd()) if file.startswith("readings_") and file.endswith(".parquet")]
# readings_files.remove("readings_5000.parquet")
test_file_index = choice(range(len(joined_files)))
joined_test_file = joined_files[test_file_index]
joined_files.remove(joined_test_file)
readings_test_file = readings_files[test_file_index]
readings_files.remove(readings_test_file)
# %%
input_voltages = keras.Input(shape = (5000, 12), name = "voltages")
# Add a 1D convolutional layer
conv1 = (Conv1D(
    filters=32, 
    kernel_size=20,
    strides = 3,  
    activation='relu'))(input_voltages)
# Add a max pooling layer
pool1 = (MaxPooling1D(pool_size=4))(conv1)
# Add another 1D convolutional layer
conv2 = (Conv1D(filters=32, kernel_size=20, activation='relu'))(pool1)
# One more max pooling layer
pool2 = (MaxPooling1D(pool_size=4))(conv2)
# Add another 1D convolutional layer
conv3 = (Conv1D(filters=32, kernel_size=20, activation='relu'))(pool2)
# One more max pooling layer
pool3 = (MaxPooling1D(pool_size=4))(conv3)
# Flatten the output before passing it to the dense layers
flatten = (Flatten())(pool3)
other_inputs = keras.Input(shape = (111,), name = "other")
concatenated_layer = keras.layers.concatenate([flatten, other_inputs])
dense1 = (Dense(64, activation='relu'))(concatenated_layer)
dense2 = (Dense(42, activation='relu'))(dense1)
denseout = (Dense(1, activation='sigmoid'))(dense2)
model = keras.Model(
    inputs=[input_voltages, other_inputs],
    outputs=[denseout],
)
# Adjust the output layer based on your problem
# Print the model summary to check the architecture
# model.summary()
# Compile the Model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy')  # Adjust the optimizer and loss function based on your problem
#%%
i = 0
for joined, readings in zip(joined_files, readings_files):
    # print(joined, pd.read_parquet(joined).shape)
    # print(readings, pd.read_parquet(readings).shape)
    X=None
    i += 1
    X, X2, y = load_and_preprocess_data(joined, readings, y_var = y_var)
    # Train the Model
    print(joined,readings, f"starting {i}/{len(joined_files)}", sep = "\n")
    model.fit(
        {"voltages": X,
         "other"   : X2}, 
        y, 
        epochs=1, 
        batch_size=32, 
        validation_split=0.2)  # Adjust the number of epochs, validation split, and batch size as needed



# %% # Defining y_pred variables
X_test, X2_test, y_test = load_and_preprocess_data(joined_test_file, readings_test_file, y_var = y_var)

y_pred_probs = model.predict_on_batch(
    [X_test,
     X2_test])
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

# Display the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Confusion Matrix for {y_var}")
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# %%