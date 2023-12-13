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
import tensorflow as tf
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
y_var = "SB"
possible_diagnoses = pd.read_csv(r"C:\Users\David\Documents\David BYU-Idaho\Fall 2023\DS 499\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]

# %% # Read in the parquet files as the X variable

file_path = "readings_test_" # change to "readings_test_" for doing a small portion of the data to work with before the real model. Use "readings" to run the real thing. It just takes a lot longer
for i in ("0","50000","100000","150000","200000","250000","300000","350000","400000","450000","500000"):
    print(i)
    if os.path.exists(f"{file_path}{i}.parquet"):
        print(i)
        if i == "0":
            dat_df = pd.read_parquet(f"{file_path}{i}.parquet")
        else:
            dat_df = pd.concat([dat_df, pd.read_parquet(f"{file_path}{i}.parquet")])
# %% # Read in the records df
y = pd.read_parquet("records.parquet")[y_var]



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
X_test, X_train, y_test, y_train = \
    train_test_split(
        X, 
        y, 
        test_size = 0.8, 
        random_state = my_seed)


# %% # 
model = Sequential()
# Add a 1D convolutional layer
model.add(Conv1D(
    filters=32, 
    kernel_size=20,
    strides = 3,  
    activation='relu', 
    input_shape=(5000, 12)))
# Add a max pooling layer
model.add(MaxPooling1D(pool_size=4))
# Add another 1D convolutional layer
model.add(Conv1D(filters=32, kernel_size=20, activation='relu'))
# One more max pooling layer
model.add(MaxPooling1D(pool_size=4))
# Add another 1D convolutional layer
model.add(Conv1D(filters=32, kernel_size=20, activation='relu'))
# One more max pooling layer
model.add(MaxPooling1D(pool_size=4))
# Flatten the output before passing it to the dense layers
model.add(Flatten())
# Add one or more dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Adjust the output layer based on your problem
# Print the model summary to check the architecture
# model.summary()

# Compile the Model
model.compile(
    optimizer='adam',
    loss='mean_squared_error')  # Adjust the optimizer and loss function based on your problem

# %%
# Train the Model
model.fit(
    X_train, 
    y_train, 
    epochs=5, 
    batch_size=32, 
    validation_split=0.1)  # Adjust the number of epochs and batch size as needed


# %%
y_pred_probs = model.predict(X_test)
y_pred_rounded = np.round(y_pred_probs)
# %%
# Create a confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred_rounded)

# Display the confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# %%