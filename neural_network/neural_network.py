"""
Module containing the code responsible for the neural network model and everything that comes with it
"""
import numpy as np
from typing import Callable
from neural_network.layer import Layer


class NeuralNetwork:
    def __init__(self, x_train, x_eval, y_train, y_eval):
        self.layers: list[Layer] = []

    def add_layer(self, n_inputs: int, n_neurons: int, activation_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.layers.append(Layer(n_inputs, n_neurons, activation_function))

    def train(self) -> None:
        pass
