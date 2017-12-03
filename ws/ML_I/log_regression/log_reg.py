import numpy as np
import pandas as pd

def init_data():
    data = pd.read_csv('binary.csv')
    X = data[['gre', 'gpa', 'rank']].values
    y = data[['admit']].values.flatten()
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X = (X - means) / stds
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)

    return np.sum(target * scores - np.log(1 + np.exp(scores)))


def grad(features, target, predictions):
    diff = target - predictions

    return np.dot(features.T, diff)


def logistic_regression(features, target, num_steps, learning_rate):
    features = np.hstack(
        (np.ones((features.shape[0], 1)), features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        gradient = grad(features, target, predictions)
        weights += learning_rate * gradient

    return weights


if __name__ == '__main__':
    features, labels = init_data()
    weights = logistic_regression(features, labels, num_steps = 100000, learning_rate = 0.0001)
    print(weights)
