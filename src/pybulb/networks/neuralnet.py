import numpy as np


class Node:
    def __init__(self, activation='sigmoid'):
        self.activation = activation
        self.input = 0
        self.output = 0
        self.delta = 0

    def activate(self, x):
        self.input = x
        activation = globals()[self.activation]
        self.output = activation(x)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.output * (1 - self.output)


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)
        self.outputs = np.zeros(output_size)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.outputs = self.sigmoid(self.z)
        return self.outputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.outputs * (1 - self.outputs)

    def backward(self, output_error, learning_rate):
        # Calculate gradient
        delta = output_error * self.sigmoid_derivative()
        input_error = np.dot(delta, self.weights.T)
        weights_error = np.dot(self.inputs.T, delta)

        # Update weights and biases
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * delta.sum(axis=0)

        return input_error


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        error = y - output
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                self.backward(xi.reshape(1, -1), yi.reshape(1, -1), learning_rate)

    def predict(self, X):
        return self.forward(X)
