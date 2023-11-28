"""
Module containing the code responsible for the neural network model and everything that comes with it
"""
import numpy as np
from typing import Callable
from neural_network.layer import Layer


class NeuralNetwork:
    def __init__(self, train_x, train_y, eval_x, eval_y):
        self.train_X = train_x
        self.train_y = train_y
        self.eval_X = eval_x
        self.eval_y = eval_y
        self.layers: list[Layer] = []

    def __str__(self):
        n_layers = len(self.layers)
        s = f'Neural network model.\nNumber of layers: {n_layers}'
        for i in range(n_layers):
            s += f'\nNumber of neurons in the {i+1} layer: {np.size(self.layers[i].outputs, 1)}'
        return s

    def add_layer(self, n_inputs: int, n_neurons: int, activation_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.layers.append(Layer(n_inputs, n_neurons, activation_function))

    def train(self) -> None:
        pass
