# Project of neural network for the "Introduction to Artificial Intelligence" course

# Task description
Implement a simple neural network that predicts the class of iris plant basing on its parameters.
Use 2 hidden layers. The layers should be fully connected with RELu activation function, except for the output layer, which should use softmax function.
No regularisation or optimalisation are needed. No batching is needed. 

# Installation
The project was developed using poetry. You can install it using pip:
```
pip install poetry
```
To install the dependencies, run:
```
poetry install
```
To run the program, run:
```
poetry run app
```

# Dataset
The dataset is available in the file `Iris.csv`. The dataset contains 150 samples of iris plants.
Each sample has 4 parameters: sepal length, sepal width, petal length, petal width. The samples are divided into 3 classes: Iris Setosa, Iris Versicolour, Iris Virginica.

# Output
The output of the program should be the accuracy of the network on the test set.
The accuracy is the number of correctly classified samples divided by the number of all samples.
Beside that the program should print the plot of the loss function and accuracy value during the training process.

