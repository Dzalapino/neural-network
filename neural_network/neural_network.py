"""
Module containing the code responsible for the neural network model and its training and evaluation.
The neural network model is a list of layers. Each layer is an object of the Layer class.
The neural network model is trained using the train method. The evaluation is performed using the evaluate method.
"""

import numpy as np
from typing import Callable
from neural_network.layer import Layer


def calculate_loss(expected_values: np.ndarray, predicted_values: np.ndarray):
    """
    Method for calculation loss function for categorical cross entropy
    :param expected_values: Expected values (labels)
    :param predicted_values: Predicted values (outputs from the last layer)
    :return: Loss function value for categorical cross entropy
    """
    return -1 * np.sum(expected_values * np.log(predicted_values + 1e-15))


class NeuralNetwork:
    """
    Class representing the neural network model
    """
    def __init__(self):
        self.layers: list[Layer] = []
        self.loss = 0.
        self.total_loss = 0.
        self.avg_loss = 0.

    def add_layer(self, n_inputs: int, n_neurons: int, activation_function: Callable[[np.ndarray], np.ndarray],
                  activation_derivative: Callable[[np.ndarray], np.ndarray] | None) -> None:
        """
        Method for adding a new layer to the neural network model
        :param n_inputs: Number of inputs to the layer (number of neurons in the previous layer)
        :param n_neurons: Number of neurons in the layer (number of outputs from the layer)
        :param activation_function: Activation function for the layer (e.g. sigmoid, relu, softmax)
        :param activation_derivative: Derivative of the activation function for the layer (e.g. sigmoid_derivative,
        :return: None
        """
        self.layers.append(Layer(n_inputs, n_neurons, activation_function, activation_derivative))

    def train(self, train_x: np.ndarray, train_y: np.ndarray, learning_epochs=100,
              learning_rate=0.005, if_print=False) -> None:
        """
        Method for training the neural network model on the given training data set.
        The training is performed using the gradient descent method.
        :param train_x: Training data set features (inputs)
        :param train_y: Training data set labels (expected values)
        :param learning_epochs: Learning epochs for the gradient descent method
        :param learning_rate: Learning rate for the gradient descent method
        :param if_print: If True, the method will print the loss after each epoch
        :return: None
        """
        print(train_x)
        print(train_y)
        if if_print:
            print('Starting training the model...')
        for epoch in range(learning_epochs):
            # Shuffle the training features and labels before each epoch
            indices = np.arange(len(train_x))
            np.random.shuffle(indices)
            print(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]

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

            self.avg_loss = self.total_loss / np.size(train_x, axis=0)
            if if_print:
                print(f'  Total Loss after epoch {epoch}: {self.total_loss}\n  Avg loss after epoch {epoch}: {self.avg_loss}')

    def evaluate(self, eval_x: np.ndarray, eval_y: np.ndarray) -> None:
        """
        Method for evaluating the model on the given evaluation data set (not used for training).
        :param eval_x: Evaluation data set features (inputs)
        :param eval_y: Evaluation data set labels (expected values)
        :return: None
        """
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

    def __str__(self):
        n_layers = len(self.layers)
        s = f'Neural network with {n_layers} layers:'
        for i in range(n_layers):
            s += f'\nLayer {i+1}:\n'
            s += str(self.layers[i])
        return s
