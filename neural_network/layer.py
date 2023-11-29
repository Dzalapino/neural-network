"""
Module responsible for the layers creation and management.
It stores all neurons weights, biases and calculates outputs based on provided activation function
"""
import numpy as np
from typing import Callable


class Layer:
    def __init__(self, n_inputs, n_neurons,
                 activation_function: Callable[[np.ndarray], np.ndarray]
                 ):
        # Init weights using He initialization. Normal(mean, standard_deviation)
        self.weights: np.ndarray = np.random.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_neurons))
        # Init biases and outputs
        self.biases: np.ndarray = np.zeros((1, n_neurons))
        self.outputs: np.ndarray = np.empty((1, n_neurons))
        # Pass activation function
        self.activation_function: Callable[[np.ndarray], np.ndarray] = activation_function

    def __str__(self):
        return (f"Layer:\nweights: {self.weights}\nbiases: {self.biases}\n"
                f"outputs: {self.outputs}\nactivation function: {self.activation_function}")

    def forward_pass(self, inputs) -> None:
        # Calculate the outputs and pass them to the activation function
        self.outputs = self.activation_function(np.dot(inputs, self.weights) + self.biases)
