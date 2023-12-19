"""
Module containing the code responsible for the neural network model and everything that comes with it
"""

import numpy as np
from typing import Callable
from neural_network.layer import Layer


# Method for calculation loss function for categorical cross entropy
def calculate_loss(expected_values: np.ndarray, predicted_values: np.ndarray):
    return -1 * np.sum(expected_values * np.log(predicted_values + 1e-15))


# Neural network class that stores all layers and performs forward and backward propagation
class NeuralNetwork:
    def __init__(self):
        self.layers: list[Layer] = []
        self.loss = 0.
        self.total_loss = 0.
        self.avg_loss = 0.

    def __str__(self):
        n_layers = len(self.layers)
        s = f'Neural network with {n_layers} layers:'
        for i in range(n_layers):
            s += f'\nLayer {i+1}:\n'
            s += str(self.layers[i])
        return s

    def add_layer(self, n_inputs: int, n_neurons: int, activation_function: Callable[[np.ndarray], np.ndarray],
                  activation_derivative: Callable[[np.ndarray], np.ndarray] | None) -> None:
        """
        Method for adding a new layer to the neural network model
        :param n_inputs: number of inputs to the layer
        :param n_neurons: number of neurons in the layer
        :param activation_function: activation function for the layer
        :param activation_derivative: derivative of the activation function for the layer
        :return:
        """
        self.layers.append(Layer(n_inputs, n_neurons, activation_function, activation_derivative))

    def train(self, train_x: np.ndarray, train_y: np.ndarray, learning_epochs=100,
              learning_rate=0.005, if_print=False) -> None:
        print(train_x)
        print(train_y)
        if if_print:
            print('Starting training the model...')
        for epoch in range(learning_epochs):
            if if_print:
                print(f'  epoch {epoch}:')
            for train_sample, y in zip(train_x, train_y):
                if if_print:
                    print('\n    Forward propagation...')
                # Forward propagation
                self.layers[0].forward_pass(train_sample)  # Feed first layer with inputs from the data set
                for i in range(1, len(self.layers)):
                    # Feed next layers with activated outputs from the previous layers
                    self.layers[i].forward_pass(self.layers[i-1].get_activated_outputs())

                # Calculate categorical cross entropy loss
                self.loss = calculate_loss(y, self.layers[-1].get_activated_outputs())
                self.total_loss += self.loss

                if if_print:
                    print(f'    Loss after forward pass: {self.loss}\n    Backward propagation...')
                # Backward propagation
                # Calculate the derivative of the loss function with respect to the last layer outputs
                loss_dwrt_output = self.layers[-1].get_activated_outputs() - y
                # Calculate the derivative of the loss function with respect to the last layer weights
                loss_dwrt_weights = np.zeros(np.shape(self.layers[-1].weights))
                for neuron in range(np.size(self.layers[-1].weights, axis=1)):
                    for activation in range(np.size(self.layers[-2].get_activated_outputs())):
                        loss_dwrt_weights[activation, neuron] = (
                            (self.layers[-2].get_activated_outputs()[activation] * loss_dwrt_output[neuron])
                        )
                loss_dwrt_biases = np.array(loss_dwrt_output)

                # Update last layer weights and biases
                self.layers[-1].weights -= learning_rate * loss_dwrt_weights
                self.layers[-1].biases -= learning_rate * loss_dwrt_biases

                # Propagate the error to the previous layers
                for layer in range(len(self.layers) - 2, -1, -1):
                    # Calculate the derivative of the loss function with respect to the outputs
                    new_loss_dwrt_output = np.zeros(np.size(self.layers[layer].outputs))
                    for output in range(np.size(self.layers[layer].outputs)):
                        for d_output in range(np.size(loss_dwrt_output)):
                            new_loss_dwrt_output[output] += (
                                    loss_dwrt_output[d_output] * self.layers[layer + 1].weights[output, d_output]
                            )
                    new_loss_dwrt_output *= self.layers[layer].get_derivative_of_outputs()
                    loss_dwrt_output = new_loss_dwrt_output

                    # Calculate the derivative of the loss function with respect to the weights
                    loss_dwrt_weights = np.zeros(np.shape(self.layers[layer].weights))
                    if layer > 0:  # Use activated outputs from the previous layer
                        for neuron in range(np.size(self.layers[layer].outputs)):
                            for weight in range(np.size(loss_dwrt_weights, axis=0)):
                                loss_dwrt_weights[weight, neuron] = (
                                        self.layers[layer-1].get_activated_outputs()[weight]
                                        * new_loss_dwrt_output[neuron]
                                )
                    else:  # Use train sample
                        for neuron in range(np.size(self.layers[layer].outputs)):
                            for weight in range(len(train_sample)):
                                loss_dwrt_weights[weight, neuron] = (
                                        train_sample[weight] * new_loss_dwrt_output[neuron]
                                )
                    # Calculate the derivative of the loss function with respect to the biases
                    loss_dwrt_biases = new_loss_dwrt_output

                    # Update the hidden layer weights and biases
                    self.layers[layer].weights -= learning_rate * loss_dwrt_weights
                    self.layers[layer].biases -= learning_rate * loss_dwrt_biases

            # Shuffle the training features and labels after each epoch
            indices = np.arange(len(train_x))
            np.random.shuffle(indices)
            print(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]

            self.avg_loss = self.total_loss / np.size(train_x, axis=0)
            if if_print:
                print(f'  Total Loss after epoch {epoch}: {self.total_loss}\n  Avg loss after epoch {epoch}: {self.avg_loss}')

    def evaluate(self, eval_x: np.ndarray, eval_y: np.ndarray) -> None:
        total_loss = 0
        for evaluation_sample, y in zip(eval_x, eval_y):
            # Forward propagation
            self.layers[0].forward_pass(evaluation_sample)  # Feed first layer with inputs from the data set
            for i in range(1, len(self.layers)):
                # Feed next layers with activated outputs from the previous layers
                self.layers[i].forward_pass(self.layers[i - 1].get_activated_outputs())

            # Calculate categorical cross entropy loss
            loss = calculate_loss(y, self.layers[-1].get_activated_outputs())
            total_loss += loss
            print(f'\nFor the evaluation sample: {evaluation_sample} = {y}')
            np.set_printoptions(precision=3, suppress=True)
            print(f'Model estimation: {self.layers[-1].get_activated_outputs()}')
            print(f'Loss: {loss}')
        print(f'\nAvg loss of the model: {total_loss/len(eval_x)}')
