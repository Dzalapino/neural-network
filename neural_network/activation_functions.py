"""
Module containing all activation functions that may be used in the neural net layers
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


def softmax(output: np.ndarray) -> np.ndarray:
    """
    The softmax function, also known as normalized exponential function,
    converts a vector of K real numbers into a probability distribution of K possible outcomes.
    :param output: output: output values from the neurons after calculating the weighted sum of inputs
    :return: transformed array of outputs
    """
    # Optimization for the big numbers
    exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
