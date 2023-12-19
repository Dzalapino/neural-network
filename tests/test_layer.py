from unittest import TestCase
import numpy as np


class TestLayer(TestCase):
    def setUp(self):
        from neural_network.layer import Layer
        from neural_network.activation_functions import relu, relu_derivative
        self.layer = Layer(n_inputs=2, n_neurons=3, activation_function=relu,
                           activation_derivative=relu_derivative)

    def test_init(self):
        self.assertEqual(self.layer.weights.shape, (2, 3))
        self.assertEqual(self.layer.biases.shape, (3,))
        self.assertEqual(self.layer.outputs.shape, (3,))
        self.assertEqual(self.layer.activation_function.__name__, 'relu')
        self.assertEqual(self.layer.activation_derivative.__name__, 'relu_derivative')

    def test_forward_pass(self):
        self.layer.weights = np.array([[0.1, 0.2, 0.3],
                                       [0.4, 0.5, 0.6]])
        self.layer.biases = np.array([0.1, 0.2, 0.3])
        self.layer.forward_pass(np.array([0.1, 0.2]))
        self.assertTrue(np.allclose(self.layer.outputs, [0.19, 0.32, 0.45]))

    def test_get_activated_outputs(self):
        self.layer.outputs = np.array([0.19, -0.32, 0.45])
        self.assertTrue(np.allclose(self.layer.get_activated_outputs(), [0.19, 0, 0.45]))

    def test_get_derivative_of_outputs(self):
        self.layer.outputs = np.array([0.19, -0.32, 0.45])
        self.assertTrue(np.allclose(self.layer.get_derivative_of_outputs(), [1, 0, 1]))
