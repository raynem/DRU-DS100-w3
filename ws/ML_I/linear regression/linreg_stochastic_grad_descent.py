#stochastic gradient descent use

import pandas as pd
import numpy as np

def compute_error(y, w, X):
    error = 0
    N = len(X)
    error = (np.dot(X,w)-y.T)**2
    return np.sum(error) / N

def stochastic_gradient_step(X, y, w, train_ind, eta=0.1):
    grad = (2/len(X)) * X[train_ind,:] * (np.dot(X[train_ind,:],w) - y[train_ind,:])
    return  w - eta * grad


def stochastic_gradient_descent(X, y, w_init, max_iter = 10000):
    w = w_init
    errors = []
    np.random.seed(42)

    for i in range(max_iter):
        random_ind = np.random.randint(X.shape[0])
        w = stochastic_gradient_step(X, y, w, random_ind)
        errors.append(compute_error(y, w, X))

    return w, errors


def linear_regression():
    # init dataset
    adver_data = pd.read_csv('advertising.csv')
    X = adver_data[['TV', 'radio', 'newspaper']].values
    y = adver_data[['sales']].values

    # matrix scaling
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X = (X - means) / stds

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    w_init = np.zeros(X.shape[1])

    print(
        'Start learning at w = {0}, error = {1}'.format(
            w_init,
            compute_error(y, w_init, X)
        )
    )

    w, errors = stochastic_gradient_descent(X, y, w_init)
    print(
        'End learning at w = {0}, error = {1}'.format(
            w,
            compute_error(y, w, X)
        )
    )



if __name__ == '__main__':

    linear_regression()