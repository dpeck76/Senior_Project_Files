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

# %% # Choose y-variable
y_var = "AFIB"
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]

# %% # Read in the parquet files as the X variable
file_path = "readings_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","5000","10000","15000","20000","25000","30000","35000","40000","45000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            dat_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            dat_df = pd.concat([dat_df, pd.read_parquet(f"{file_path}{i}.parquet")])
# %% # Read in the additional parquet file for the other x variables
file_path = "joined_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","5000","10000","15000","20000","25000","30000","35000","40000","45000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            joined_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            joined_df = pd.concat([joined_df, pd.read_parquet(f"{file_path}{i}.parquet")])
# %%
columns_to_keep = ["Age", "Sex_M"]
type = ["I",  "II","III","aVR", 
        "aVL","aVF","V1", "V2", 
        "V3", "V4", "V5", "V6"]
for i in type:
    columns_to_keep.append(f"{i}_std_elapsed_time")
for i in type:
    columns_to_keep.append(f"{i}_median_elapsed_time")
X2 = joined_df[columns_to_keep].fillna(0)


# %% # Read in the records df
y = pd.read_parquet("records_0.parquet")[y_var]

# %% # Pivot the data and convert to numpy 3d array
X = np.transpose(
        np.array(
            dat_df\
                .pivot(
                    index = "Record", 
                    columns = "Type", 
                    values = "val")\
                .to_numpy()\
                .tolist())\
                .reshape((4999, 12, -1))
            , (0, 2, 1))
# %% # Train test split
my_seed = 40
X_train, X_test, X2_train, X2_test, y_train, y_test = \
    train_test_split(
        X, 
        X2,
        y, 
        test_size = 0.2, 
        random_state = my_seed)
# %% # 
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

other_inputs = keras.Input(shape = (len(columns_to_keep),), name = "other")

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
optimizer = keras.optimizers.Adam(learning_rate=0.05)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy')  # Adjust the optimizer and loss function based on your problem

# %%
# Train the Model
model.fit(
    {"voltages": X_train,
     "other": X2_train}, 
    y_train, 
    epochs=15, 
    batch_size=32, 
    validation_split=0.15)  # Adjust the number of epochs and batch size as needed



# %% # Defining y_pred variables
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
      f"F1:        {F1:.7f}")

# %% # Create a confusion matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Confusion Matrix for {y_var}")
plt.show()

# %% # 
# from keras.utils import plot_model
# # import pydotplus
# # plot_model(model, "multi_input_and_output_model.png", show_shapes=True, expand_nested=True, dpi=300,dotprog=pydotplus)
# plot_path = 'keras_model.png'
# plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
# model.summary()
# %%
