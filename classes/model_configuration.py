'''
Model configuration functions
'''

import numpy as np
from sklearn import metrics
import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from helpers.logger import LoggerHelper, logging

LoggerHelper.init_logger()
logger = logging.getLogger(__name__)

class ModelConfiguration():

    def __init__(self):
        pass

    def early_stopping_callback(self, monitor = "val_loss", mode = "min", patience = 10, min_delta = 1e-4):
        ''' Function to stop model training when a monitored metric stops improving.
        The objective is to avoid overfitting and to improve training efficiency.

        Parameters (input):
            monitor: metric to be monitored
                "val_loss", "loss", "accuracy", "val_accuracy
            mode (str): ("min"/"max") if the minimum or maximum of the metric is wanted to reach
            patience (int): number of epochs with no improvement before stopping the training

        Returns:
            Early stopping callback
        '''

        logger.info("early_stopping_callback(): Define callback to stop training")
        logger.info("early_stopping_callback(): Monitored metric: %s" %monitor)

        early_stopping = EarlyStopping(monitor = monitor, mode = mode, verbose = 1, patience = patience, min_delta = min_delta)
        return early_stopping

    def model_checkpoint_callback(self, model_path, monitor = "val_accuracy", mode = "max", save_best_only = True):
        ''' Function to save the model or model weights.
        It is used to save the best model and avoid losing its characteristics in case of
        overfitting or performance degradation

        Parameters (input):
            filepath (str): name or path where the model is to be saved.
            monitor: metric to be monitored
                "val_loss", "loss", "accuracy", "val_accuracy
            mode (str): ("min"/"max") if the minimum or maximum of the metric is wanted to reach
            save_best_only (boolean): boolean indicating whether the best model is saved according to the monitored metric

        Returns:
            Model checkpoint callback
        '''

        logger.info("model_checkpoint_callback(): Define callback to save the best model")
        logger.info("model_checkpoint_callback(): Monitored metric %s" %monitor)

        model_checkpoint = ModelCheckpoint(model_path, monitor = monitor, mode = mode,
                                           verbose = 1, save_best_only = save_best_only)
        return model_checkpoint

    def model_compilation(self, model, loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"]):
        ''' Function to configure the model for training.
        Characteristics for training the model are defined.

        Parameters (input):
            optimizer (str): optimisation algorithm to minimise the loss function.
            loss (str): loss function. This function is used to measure the difference between
                        the predictions made by the model and the real labels during training.
            metrics (list): List of metrics to be evaluated during training and testing.

        Returns:
            The model
        '''

        logger.info("model_compilation(): Model configuration for the training")

        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        return model

    def model_fitting(self, model, X_train, y_train, X_test, y_test, callback_list, epochs = 50, batch_size = 64):
        ''' Function to train the model during the epochs.
        The model learns from the X_train data with y_train labels. During this training,
        the weights are adjusted to minimise the loss function.
        The weights are updated using an optimisation model.

        Parameters (input):
            model (model)
            X_train (tensor): training data with structure (n_series, n_timestamps, n_features)
            y_train (array): array of the labels of training data
            X_test (tensor): test data
            y_test (array): array of the labels of test data
            epochs (int): number of epochs, number of iterations of all the training data.
            batch_size (int, power of 2): number of training samples or instances used to update the weights in each iteration.
            callback_list (list): list of callbacks.

        Returns:
            Trained model
        '''

        logger.info("model_fitting(): Model training")

        history = model.fit(X_train, y_train, validation_data = (X_test, y_test),
                            batch_size = batch_size, epochs = epochs, callbacks = callback_list, verbose = 1)

        return history

    def model_evaluation(self, model, X, y, X_test, y_test):
        ''' Function to evaluate the model on the entire dataset and on the test set.

        Parameters (input):
            model (model)
            X (tensor): dataset with structure (n_series, n_timestamps, n_features)
            y (array): array of the labels of the dataset
            X_test (tensor): test data
            y_test (array): array of the labels of test data
        '''
        
        logger.info("model_evaluation(): Evaluation of the model with metrics %s" %model.metrics_names)

        model_metrics_data = model.evaluate(X, y, return_dict = True)
        for metric, value in model_metrics_data.items():
            logger.info("model_evaluation(): %s of the dataset: %s" %(metric, value))

        model_metrics_test = model.evaluate(X_test, y_test, return_dict = True)
        for metric, value in model_metrics_test.items():
            logger.info("model_evaluation(): %s of the test data: %s" %(metric, value))

        probabilities = model.predict(X_test)

# Select the class with the highest probability
        y_pred = np.argmax(probabilities, axis = 1)
        result = np.zeros(len(y_test))
        for i in range(y_test.shape[0]):
            pos = np.where(y_test[i] == 1)[0]
            if pos.size > 0:  # If '1' is found in the row
                result[i] = pos[0]


        tp0,fp01,fp02,fp10,tp1,fp12,fp20,fp21,tp2  = metrics.confusion_matrix(result,y_pred).ravel()
        metrics_dict = {
                    "accuracy": metrics.accuracy_score(result,y_pred),
                    "recall": metrics.recall_score(result,y_pred,average='macro'),
                    "f1_score": metrics.f1_score(result,y_pred,average='macro'),
                    "precision": metrics.precision_score(result,y_pred,average='macro'),
                    "precision_electrical": metrics.precision_score(result,y_pred,average=None)[0],
                    "precision_electrical": metrics.precision_score(result,y_pred,average=None)[0],
                    "precision_mechanical": metrics.precision_score(result,y_pred,average=None)[1],
                    "precision_not_anomalous": metrics.precision_score(result,y_pred,average=None)[2],
                    "recall_electrical": metrics.recall_score(result,y_pred,average=None)[0],
                    "recall_mechanical": metrics.recall_score(result,y_pred,average=None)[1],
                    "recall_not_anomalous": metrics.recall_score(result,y_pred,average=None)[2],
                    "f1_score_electrical": metrics.f1_score(result,y_pred,average=None)[0],
                    "f1_score_mechanical": metrics.f1_score(result,y_pred,average=None)[1],
                    "f1_score_not_anomalous": metrics.f1_score(result,y_pred,average=None)[2],
                    "tp0": int(tp0),
                    "fp01": int(fp01),
                    "fp02": int(fp02),
                    "fp10": int(fp10),
                    "tp1": int(tp1),
                    "fp12": int(fp12),
                    "fp20": int(fp20),
                    "fp21": int(fp21),
                    "tp2": int(tp2),
                    }

        return metrics_dict

    def model_prediction(self, model, X):
        ''' Function to compute predictions.

        Parameters (input):
            model
            X (tensor): dataset for which predictions are to be calculated.

        Returns:
            Class for each observation
        '''

        logger.info("model_predictions(): Compute predictions")

        # Returns the probability an observation belongs to each class.
        probabilities = model.predict(X)

        # Select the class with the highest probability
        predictions = np.argmax(probabilities, axis = 1)

        return predictions









