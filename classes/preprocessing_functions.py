'''
Data preprocessing functions for model training
'''

import os
import zipfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.utils import pad_sequences, to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from helpers.logger import LoggerHelper, logging

# Load logger
LoggerHelper.init_logger()
logger = logging.getLogger(__name__)

def read_data(data_path, indicator_list):
    ''' Function for reading and loading data.

    Parameters (input):
        data_path (str): data path
        indicator_list (list): list of indicators te be read from the files

    Returns:
        X: a tensor contianing the data of each file in a list
        Y: list of labels
    '''

    logger.info("read_data(): Read data from path %s" %data_path)
    logger.info("read_data(): %d variables are read: %s" %(len(indicator_list), indicator_list))

    labels = {
        "not_anomalous": "not anomalous",
        "mechanical_anomalies": "mechanical anomaly",
        "electrical_anomalies": "electrical anomaly"
    }
    
    # Folder list
    folders = os.listdir(data_path)

    # Initialize lists to store data
    X = list()
    Y = list()
    Z = list()
    # Loop de folders
    for folder in labels.keys():

        folder_path = os.path.join(data_path, folder)
        days = os.listdir(folder_path)

        # The label of the data is the name of the folder (ELECTRICAL ANOMALIES / MECHANICAL ANOMALIES / NOT ANOMALOUS)
        y = labels[folder]

        for day in days:
            day_path = os.path.join(folder_path, day)
            files = os.listdir(day_path)

            # Loop for the files in the day
            for file in files:

                # File path
                file_path = os.path.join(day_path, file)

                # Unzip the file y read it as DataFrame
                zip_file = zipfile.ZipFile(file_path, 'r')
                filename = zip_file.namelist()[0]
                df = pd.read_csv(zip_file.open(filename), delimiter = ";")

                # Append to the list the data and the label
                X.append(df[indicator_list].to_numpy())
                Y.append(y)
                Z.append(file)


    logger.info("read_data(): Number of files read %d" %len(X))

    return X, Y, Z

def add_padding(X, indicator_list):
    ''' Function to add paddind to the time series, i.e. zeros or nan values to incomplete series

    Parameters (input):
        X (list): data read from the files
        indicator_list (list): list of indicators to be considered as features

    Returns:
        X_pad: data (tensor) after adding padding
    '''

    logger.info("add_padding(): Matching the length of the time series adding padding")

    n_features = len(indicator_list)

    # X_pad = pad_sequences(X, padding = "post", dtype = "float64", value = np.full((n_features,), np.nan))
    X_pad = pad_sequences(X, padding = "post", dtype = "float64", value = np.full((n_features,), 0))

    return X_pad

def encode_response_variable(y):
    ''' Converts a class vector (strings) to binary class matrix (one-hot ecoding).

    Parameters (input):
        y (array): array containing string labels

    Returns:
        y_encoded: array containing one hot encoding
    '''

    # Encode class values as integers
    encoder = LabelEncoder()

    # Encode strings into integers
    y_labels = encoder.fit_transform(y)

    # One-hot encoding
    y_encoded = to_categorical(y_labels, num_classes = len(set(y)))
    print(y_encoded)

    return y_encoded

def split_data(X, Y, Z, test_size = 0.2):
    ''' Function to split data into training set and test set.

    Parameters (input):
        X (tensor): dataset
        Y (array): labels
        test_size (float): percentage of data for the test set

    Returns:
        X_train, y_train: data for the trainng set (data and response variable)
        X_test, y_test: data for the test set (data and response variable)
    '''

    #logger.info("split_data(): Split data into training and test sets (%f)" %test_size)

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, Y, Z, test_size = 0.2, random_state = 123)
    return X_train, X_test, y_train, y_test,z_train, z_test

def plot_model_history(history, output_path):
    ''' Function to plot model metrics during training epochs.

    Parameters (input):
        history: model history
        output_path (str): path where to store the plot
    '''

    # Initialize plot
    fig, ax = plt.subplots(figsize = (16,10))

    # Accuracy evolution during the epochs in the training set and validation set
    p1 = ax.plot(history.history['accuracy'], color = "b", label = "Accuracy of training data")
    p2 = ax.plot(history.history['val_accuracy'], color = "b", linestyle = "--", label = "Accuracy of validation data")

    # Loss function evolution during the epochs in the training set and validation set
    ax2 = ax.twinx()
    p3 = ax2.plot(history.history['loss'], color = "r", label = "Loss of training data")
    p4 = ax2.plot(history.history['val_loss'], color = "r", linestyle = "--", label = "Loss of validation data")

    plt.title("Model accuracy and loss", fontsize = 18)
    ax.set_xlabel("Training epoch", fontsize = 14)
    ax.set_ylabel("Accuracy", fontsize = 14)
    ax2.set_ylabel("Loss", fontsize = 14)

    # Add labels in the legend
    lns = p1 + p2 + p3 + p4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc = 'upper right')

    # Save the plot
    path_plot = os.path.join(output_path, "model_history")
    plt.savefig(path_plot, dpi = 120)

def load_model(model_path):
    ''' Function to load the model that is stored in the path model_path

    Parameters (input):
        model_path (str): path where the model is stored

    Returns:
        The model
    '''

    logger.info("load_model(): Load the model %s" %os.path.basename(model_path))

    model = keras.models.load_model(model_path)

    logger.info("load_model(): Model structure")
    logger.info(model.summary())

    return model

def save_model(model,model_path):
    ''' Function to load the model that is stored in the path model_path

    Parameters (input):
        model_path (str): path where the model is stored

    Returns:
        The model
    '''

    logger.info("load_model(): Load the model %s" %os.path.basename(model_path))

    model = keras.models.save_model(model,model_path)

    logger.info("save_model(): Model structure")
    #logger.info(model.summary())








