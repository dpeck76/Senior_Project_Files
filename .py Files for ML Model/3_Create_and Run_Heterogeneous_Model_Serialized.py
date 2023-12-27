# %% # 1:Import needed libraries
import seaborn as sns
import os
import numpy as np
import pandas as pd
import imblearn as il
from tensorflow.keras import backend as K
from random import choice
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
##### From CSE 450 NN code:
# https://colab.research.google.com/github/byui-cse/cse450-course/blob/master/notebooks/hint_nn.ipynb#scrollTo=KlZiVE696408
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
# %% # Choose y-variable
possible_diagnoses = pd.read_csv(r"..\electrocardiogram-database-arrhythmia-study\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\ConditionNames_SNOMED-CT.csv")
pos_d = possible_diagnoses["Acronym Name"]
# Should be a list even if it is only one element long
y_var = ["SB"]
# %% # Define the function to load and preprocess the data
def load_and_preprocess_data(joined_file_path, readings_file_path, y_var = pos_d):
    # Read in both data frames - ensure that joined_file_path and readings_file_path have data from the same records
    joined_dat_df = pd.read_parquet(joined_file_path)
    readings_dat_df = pd.read_parquet(readings_file_path)
    # Create the tabular x - variable
    X2 = joined_dat_df\
        .drop(pos_d, axis = 1)\
        .drop(["Record", "diagnosis_rowwise_sum", "diagnosis_count", "Diagnoses", "Diagnoses_list"], axis = 1)\
        .fillna(0)
    # Create the y - variable
    y = joined_dat_df[y_var]
    # Create the signal-type X variable with the readings in it
    X = \
        np.transpose(
            np.array(
                readings_dat_df\
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
# %% # Randomly choose a file to be the test file
# Set a seed so that the outcome will be the same each time
my_seed = 40
# Find all of the file paths in the current working directory to the files that we wrote using "2_add_features.py" and save the "joined" file paths to one list and the "readings" file paths to another list.
joined_files = [file for file in os.listdir(os.getcwd()) if file.startswith("joined_") and file.endswith(".parquet")]
readings_files = [file for file in os.listdir(os.getcwd()) if file.startswith("readings_") and file.endswith(".parquet")]
# Pick a random index number to use as the test file:
test_file_index = choice(range(len(joined_files)))
# Assign a test file variable for the joined dataframe
joined_test_file = joined_files[test_file_index]
# Remove the test file path from the list that we will use for training
joined_files.remove(joined_test_file)
# Same process with the readings files
readings_test_file = readings_files[test_file_index]
readings_files.remove(readings_test_file)
# %% # Create the model
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
# Create a 2nd input layer for the tabular inputs
other_inputs = keras.Input(shape = (111,), name = "other")
concatenated_layer = keras.layers.concatenate([flatten, other_inputs])
dense1 = (Dense(64, activation='relu'))(concatenated_layer)
dense2 = (Dense(42, activation='relu'))(dense1)
denseout = (Dense(len(y_var), activation='sigmoid'))(dense2)
model = keras.Model(
    inputs=[input_voltages, other_inputs],
    outputs=[denseout],
)
# Compile the Model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy')  # Adjust the optimizer and loss function based on your problem
#%% # Train the model
# Train the model
i = 0
for joined, readings in zip(joined_files, readings_files):
    X=None
    i += 1
    X, X2, y = load_and_preprocess_data(joined, readings, y_var = y_var)
    # Train the Model
    print(joined,readings, f"starting {i}/{len(joined_files)}", sep = "\n")
    model.fit(
        {"voltages": X,
         "other"   : X2}, 
        y, 
        epochs=3, 
        batch_size=32, 
        validation_split=0.2)  # Adjust the number of epochs, validation split, and batch size as needed



# %% # Defining y_pred variables
X_test, X2_test, y_test = load_and_preprocess_data(joined_test_file, readings_test_file, y_var = y_var)

y_pred_probs = model.predict_on_batch(
    [X_test,
     X2_test])
# y_pred_rounded = np.round(y_pred_probs)
y_pred = y_pred_probs
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
# y_pred = np.round(y_pred_probs)

# %% # Metrics
y_pred = pd.DataFrame(y_pred)
y_pred.columns = y_var
for var in y_var:
    accuracy = accuracy_score(y_test[var], y_pred[var])
    precision = precision_score(y_test[var], y_pred[var])
    recall = recall_score(y_test[var], y_pred[var])
    F1 = f1_score(y_test[var], y_pred[var])
    print(f" {var} Accuracy:  {accuracy:.7f}\n",
        f"{var} Precision: {precision:.7f}\n",
        f"{var} Recall:    {recall:.7f}\n",
        f"{var} F1:        {F1:.7f}",)

# %% # Create a confusion matrix

for var in y_var:
    conf_matrix = confusion_matrix(y_test[var], y_pred[var])
    # Display the confusion matrix using seaborn
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", annot_kws={"size": 16})
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {var}")
    plt.show()

# %%
y_pred_probs = pd.DataFrame(y_pred_probs)
y_pred_probs.columns = y_var
from sklearn.metrics import roc_curve, auc
for var in y_var:
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test[var], y_pred_probs[var])
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {var}')
    plt.legend(loc='lower right')
    plt.show()
# %%






##########################

from plotly import express as px
def graph_it(n):
    to_graph = list(readings["val"].iloc[n])
    df = pd.DataFrame({"val": to_graph, "x": range(1, len(to_graph) + 1)})
    fig = px.line(x = "x", y = "val", data_frame = df)
    fig = fig.update_layout(
        title='',
        xaxis_title='Time (1/500 sec)',
        yaxis_title='Voltage',
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    fig = fig.update_traces(line=dict(color='red'))
    return fig

