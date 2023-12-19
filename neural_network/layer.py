"""
Module responsible for the layers creation and management.
It stores all neurons weights, biases and calculates outputs based on provided activation function
"""
import numpy as np
from typing import Callable


class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function: Callable[[np.ndarray], np.ndarray],
                 activation_derivative: Callable[[np.ndarray], np.ndarray] | None
                 ):
        # Init weights using He initialization. Normal(mean, standard_deviation)
        np.random.seed(42)  # For reproducibility
        self.weights: np.ndarray = np.random.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_neurons))
        # Init biases and outputs
        self.biases: np.ndarray = np.zeros(n_neurons)
        self.outputs: np.ndarray = np.zeros(n_neurons)
        # Pass activation function
        self.activation_function: Callable[[np.ndarray], np.ndarray] = activation_function
        self.activation_derivative: Callable[[np.ndarray], np.ndarray] = activation_derivative

    def __str__(self):
        n_neurons = len(self.outputs[0])
        s = f'Layer with {n_neurons} neurons and {self.activation_function.__name__} activation function:'
        s += f'biases: {self.biases}, outputs: {self.outputs}, activated outputs: {self.get_activated_outputs()}'
        for i in range(n_neurons):
            s += f'\nNeuron {i + 1}:\n' \
                 f'weights: {self.weights[:, i]}\n' \
                 f'bias: {self.biases[0, i]}\n' \
                 f'output: {self.outputs[0, i]}\n' \
                 f'activated output: {self.get_activated_outputs()[0, i]}\n'
        return s

    def forward_pass(self, inputs) -> None:
        # Calculate the outputs of the layer
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def get_activated_outputs(self) -> np.ndarray: return self.activation_function(self.outputs)

    def get_derivative_of_outputs(self) -> np.ndarray: return self.activation_derivative(self.outputs)
