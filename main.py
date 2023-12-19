def main():
    import matplotlib.pyplot as plt
    from neural_network.dataset import Dataset
    import neural_network.activation_functions as activation_functions
    from neural_network.neural_network import NeuralNetwork

    dataset = Dataset('resources/Iris.csv', 0.8)

    # Create the neural network model
    model = NeuralNetwork()
    model.add_layer(4, 4, activation_functions.relu, activation_functions.relu_derivative)
    model.add_layer(4, 8, activation_functions.relu, activation_functions.relu_derivative)
    model.add_layer(8, 7, activation_functions.relu, activation_functions.relu_derivative)
    model.add_layer(7, 3, activation_functions.softmax, None)

    # Train and evaluate the model
    model.train(dataset.train_X, dataset.train_y, 10, 0.003)
    model.evaluate(dataset.eval_X, dataset.eval_y)

    # Init the plot for accuracy and loss for each epoch of training
    fig, axs = plt.subplots(2)
    fig.suptitle('Model accuracy and loss')
    # Plot the accuracy
    axs[0].plot(model.accuracy_for_epoch)
    axs[0].set_ylabel('Accuracy')
    # Set the x-axis ticks to be the same as the number of epochs
    axs[0].set_xticks(range(len(model.accuracy_for_epoch)))
    # Plot the loss
    axs[1].plot(model.avg_loss_for_epoch)
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    # Set the x-axis ticks to be the same as the number of epochs
    axs[1].set_xticks(range(len(model.avg_loss_for_epoch)))
    plt.show()


if __name__ == '__main__':
    main()
