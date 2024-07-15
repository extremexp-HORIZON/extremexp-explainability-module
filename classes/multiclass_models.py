'''
Define classes for Deep Learning models for timeseries classification.
'''

import numpy as np

import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Masking, Dense, Flatten, Conv1D, MaxPooling1D, SimpleRNN, LSTM

from classes.model_configuration import ModelConfiguration

from helpers.logger import LoggerHelper, logging

# Load logger
LoggerHelper.init_logger()
logger = logging.getLogger(__name__)

class NeuralNetwork(ModelConfiguration):
    def __init__(self, n_timestamps, n_features, activation_function, units, n_classes):

        # Length of time series and number of variables/features
        self.n_timestamps = n_timestamps
        self.n_features = n_features
        self.input_shape = (n_timestamps, n_features)

        # Activation function to add non-linearity (to be able to learn more complex relationships in the data)
        self.activation_function = activation_function

        # Number of neurons in each layer
        self.units = units

        # Number of classes for multiclass classification
        self.n_classes = n_classes

        self.model = None

        super().__init__()

    def create_model(self):
        ''' Function to create the model Neural Network according to the architecture defined.
        The layers are added as indicated in self.units. The number of the layers is the length of this list.

        Linear operation are carried out and an activation function is applied to apply non-linearity.
        Applying the activation function more complex relationships in the data are learned.

        Lastly, a layer with a node and activation function "sigmoid" is added as we have
        a binary classification problem.

        Returns:
            The model
        '''

        logger.info("create_model(): Create model NEURAL NETWORK with %d layers" %len(self.units))

        # Initialize model sequence
        self.model = Sequential()

        # Input layer: input of the model
        self.model.add(Input(shape = self.input_shape))

        # Masking layer to skip the padding values during training
        self.model.add(Masking(mask_value = 0))
        # self.model.add(Masking(mask_value = np.nan))

        # Loop to add fully connected layers
        for unit in self.units:
            self.model.add(Dense(units = unit, activation = self.activation_function))

        # Flatten layer: multidimensional output into a vector
        self.model.add(Flatten())

        # Last layer for multiclass classification with the number of neurons equal to the number of classes
        # (output layer must create one output value per each class) and activation function softmax.
        # Returns the probability of an observation belonging to each class.
        self.model.add(Dense(self.n_classes, activation = "softmax"))

        logger.info("create_model(): Summary of the model")
        logger.info(self.model.summary(print_fn = logger.info))

        return self.model


class ConvolutionalNeuralNetwork(ModelConfiguration):
    def __init__(self, n_timestamps, n_features, activation_function, filters, kernel_size, pool_size, n_classes):

        # Length of time series and number of variables/features
        self.n_timestamps = n_timestamps
        self.n_features = n_features
        self.input_shape = (n_timestamps, n_features)

        # Activation function to add non-linearity (to be able to learn more complex relationships in the data)
        self.activation_function = activation_function

        # Number of filters and filter size/dimension
        self.filters = filters
        self.kernel_size = kernel_size

        # Dimension of the pooling window (how much the dimensionality is reduced)
        self.pool_size = pool_size

        # Number of classes for multiclass classification
        self.n_classes = n_classes

        self.model = None

        super().__init__()

    def create_model(self):
        ''' Function to create the model CNN (Convolutional Neural Network) according to the architecture defined.
        The architecture of a CNN consists of convolutional and pooling layers.
        The number of the layers added to the network is equal to the length of self.filters.

        Conv1D layer is applied to extract temporal patterns (on the time dimension or time axis).
        Pooling layer is applied to reduce dimensionality.
        An activation function is applied to add non-linearity.
        Finally, a layer with a node and a sigmoid activation function is added as we have
        a binary classification problem.

        Returns:
            The model
        '''

        # Initialize model sequence
        self.model = Sequential()

        # Input layer: input of the model
        self.model.add(Input(shape = self.input_shape))

        # Masking layer to skip the padding values during training
        self.model.add(Masking(mask_value = 0))
        # self.model.add(Masking(mask_value = np.nan))

        # Loop to add layers
        for filter in self.filters:

            # Convolution layer: extract characteristics from the data
            # padding = "same": we make sure that the filter is applied to all input elements
            self.model.add(Conv1D(filters = filter, kernel_size = self.kernel_size, padding = "same", activation = self.activation_function))

            # Pooling layer: reduce dimensionality
            # strides = None: non over lapping blocks
            self.model.add(MaxPooling1D(pool_size = self.pool_size, strides = None))

        # Flatten layer: multidimensional output into a vector
        self.model.add(Flatten())

        # Last layer for multiclass classification with the number of neurons equal to the number of classes
        # (output layer must create one output value per each class) and activation function softmax.
        # Returns the probability of an observation belonging to each class.
        self.model.add(Dense(self.n_classes, activation = "softmax"))

        logger.info("create_model(): Summary of the model")
        logger.info(self.model.summary(print_fn = logger.info))

        return self.model

class RecurrentNeuralNetwork(ModelConfiguration):
    def __init__(self, n_timestamps, n_features, activation_function, hidden_units, n_classes):

        # Length of time series and number of variables/features
        self.n_timestamps = n_timestamps
        self.n_features = n_features
        self.input_shape = (n_timestamps, n_features)

        # Activation function to add non-linearity (to be able to learn more complex relationships in the data)
        self.activation_function = activation_function

        # Number of neurons in each layer
        self.hidden_units = hidden_units

        # Number of classes for multiclass classification
        self.n_classes = n_classes

        self.model = None

        super().__init__()

    def create_model(self):
        ''' Function to create the model RNN (Recurrent Neural Network) according to the architecture defined.
        Recurrent neural networks allow sequence processing.
        Vanilla RNN is a Simple RNN, the simplest architecture of recurrent networks.

        The architecture consists of layers with hidden states of dimension of the units.
        At each time step (each hidden state), it takes the input of that time instant (t) and
        the combination of the outputs of previous time steps.

        It is a "Many to one" architecture: one input sequence (several time steps), one output (label 0 or 1).
        An activation function is applied to add non-linearity.
        Finally, a layer with a node and a sigmoid activation function is added as we have a binary classification problem.

        Returns:
            The model
        '''

        # Initialize model sequence
        self.model = Sequential()

        # Input layer: input of the model
        self.model.add(Input(shape = self.input_shape))

        # Masking layer to skip the padding values during training
        self.model.add(Masking(mask_value = 0))
        # self.model.add(Masking(mask_value = np.nan))

        # Loop to add layers
        for units in self.hidden_units[:-1]:

            # return_sequences = True when adding more than one layer
            self.model.add(SimpleRNN(units = units, activation = self.activation_function, return_sequences = True))

        self.model.add(SimpleRNN(units = self.hidden_units[-1], activation = self.activation_function))

        # Last layer for multiclass classification with the number of neurons equal to the number of classes
        # (output layer must create one output value per each class) and activation function softmax.
        # Returns the probability of an observation belonging to each class.
        self.model.add(Dense(self.n_classes, activation = "softmax"))

        logger.info("create_model(): Summary of the model")
        logger.info(self.model.summary(print_fn = logger.info))

        return self.model


class LongShortTermMemory(ModelConfiguration):
    def __init__(self, n_timestamps, n_features, activation_function, hidden_units, n_classes):

        # Length of time series and number of variables/features
        self.n_timestamps = n_timestamps
        self.n_features = n_features
        self.input_shape = (n_timestamps, n_features)

        # Activation function to add non-linearity (to be able to learn more complex relationships in the data)
        self.activation_function = activation_function

        # Number of neurons in each layer
        self.hidden_units = hidden_units

        # Number of classes for multiclass classification
        self.n_classes = n_classes

        self.model = None

        super().__init__()

    def create_model(self):
        ''' Function to create the model LSTM (Long Short Term Memory) according to the architecture defined.
        The LSTM network can have short-term and long-term memory. They have the ability to add or remove
        information that it is significant during the data processing.
        The cell state allows adding or removing data from the network memory.

        In order to add/remove information from the network gates are used: forget gate, update gate and
        output gate. Open gates allow information to pass through and closed doors to remove it.

        The architecture consists of hidden layers of dimension of units.
        An activation function is applied to add non linearity.
        Finally, a layer with a node and a sigmoid activation function is added as we have a binary classification problem.

        Returns:
            El modelo
        '''

        # Initialize model sequence
        self.model = Sequential()

        # Input layer: input of the model
        self.model.add(Input(shape = self.input_shape))

        # Masking layer to skip the padding values during training
        self.model.add(Masking(mask_value = 0))
        # self.model.add(Masking(mask_value = np.nan))

        # Loop to add layers
        for units in self.hidden_units[:-1]:

            # return_sequences = True when adding more than one layer
            self.model.add(LSTM(units = units, activation = self.activation_function, return_sequences = True))

        self.model.add(LSTM(units = self.hidden_units[-1], activation = self.activation_function))

        # Last layer for multiclass classification with the number of neurons equal to the number of classes
        # (output layer must create one output value per each class) and activation function softmax.
        # Returns the probability of an observation belonging to each class.
        self.model.add(Dense(self.n_classes, activation = "softmax"))

        logger.info("create_model(): Summary of the model")
        logger.info(self.model.summary(print_fn = logger.info))

        return self.model










