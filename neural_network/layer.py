import numpy as np
from typing import Callable


class Layer:
    def __init__(self, n_inputs, n_neurons,
                 activation_function: Callable[[np.ndarray], np.ndarray]
                 ):
        self.weights: np.ndarray = np.random.randn(n_inputs, n_neurons)  # Init weights
        self.biases: np.ndarray = np.zeros((1, n_neurons))  # Init biases
        self.outputs: np.ndarray = np.empty((1, n_neurons))  # Init outputs
        self.activation_function: Callable[[np.ndarray], np.ndarray] = activation_function  # Pass activation function

    def forward_pass(self, inputs) -> None:
        # Calculate the outputs and pass them to the activation function
        self.outputs = self.activation_function(np.dot(inputs, self.weights) + self.biases)
