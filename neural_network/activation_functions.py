"""
Module containing all activation functions used in the neural network model
and their derivatives for backpropagation
"""
import numpy as np


def relu(output: np.ndarray) -> np.ndarray:
    """
    The Rectified Linear Unit is the most commonly used activation function in deep learning models.
    The function returns 0 if it receives any negative input, but for any positive value x it returns that value back.
    :param output: output values from the neurons after calculating the weighted sum of inputs
    :return: transformed array of outputs
    """
    return np.maximum(0, output)


def relu_derivative(output: np.ndarray) -> np.ndarray:
    """
    Calculate the derivative of the ReLU (Rectified Linear Unit) activation function.
    :param output: output values from the neurons after applying the ReLU function
    :return: derivative array of the ReLU function outputs
    """
    return np.where(output > 0, 1, 0)


def softmax(output: np.ndarray) -> np.ndarray:
    """
    The softmax function, also known as normalized exponential function,
    converts a vector of K real numbers into a probability distribution of K possible outcomes.
    To optimize solution against big numbers we subtract the max value from all values
    :param output: output: output values from the neurons after calculating the weighted sum of inputs
    :return: transformed array of outputs
    """
    # Subtract the max value from all values to avoid overflow in exp function
    exp_values = np.exp(output - np.max(output, keepdims=True))
    return exp_values / np.sum(exp_values, keepdims=True)
