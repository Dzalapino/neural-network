# Description: This file is used to train and test the neural network.

def main():
    from neural_network.dataset import Dataset
    import neural_network.activation_functions as activation_functions
    from neural_network.neural_network import NeuralNetwork

    dataset = Dataset('resources/Iris.csv', 0.8)

    model = NeuralNetwork()
    model.add_layer(4, 10, activation_functions.relu, activation_functions.relu_derivative)
    model.add_layer(10, 8, activation_functions.relu, activation_functions.relu_derivative)
    model.add_layer(8, 3, activation_functions.softmax, None)

    model.train(dataset.train_X, dataset.train_y, 15, 0.004, True)
    model.evaluate(dataset.eval_X, dataset.eval_y)


if __name__ == '__main__':
    main()
