""" Generate random data-generating environments
"""

import numpy as np


def generate_single_layer(
    M, d, num,
):
    """
    :param M: the sparsity
    :param d: the dimension of the input
    :param num: number of gs to generate
    :return: (thetan, an, bn), where thetan has shape (num, M), an has shape (num, M, d), and bn has shape (num, M)
    """
    an = np.random.normal(0, 1 / np.sqrt(d), size=(num, M, d))
    bn = np.random.normal(0, 1 / np.sqrt(d), size=(num, M))
    thetan = np.random.normal(0, 1 / np.sqrt(M), size=(num, M))
    return (thetan, an, bn)


def generate_single_data(
    T, an, bn, thetan, act="sign",
):
    """ Generate data of length T according to the single layer generation process g specified by the parameters
    :param an: has shape (num, M, d)
    :param bn: has shape (num, M)
    :param thetan: shape (num, M)
    :param T: number of data to generate for each g
    :param act: the activation function
    :return: (X, Y), where X has shape (num, T, d), and Y has shape (num, T).
    X has standard Gaussian distribution, and the corresponding Y=g(X)
    """
    acts = {
        "sign": np.sign,
        "relu": lambda x: np.maximum(x, 0),
    }
    act = acts[act]
    (num, M, d) = an.shape
    # X: (num, T, d)
    X = np.random.normal(0, 1, (num, T, d))
    Y = (an.reshape(num, 1, M, 1, d) @
         X.reshape(num, T, 1, d, 1)).reshape(num, T, M)
    Y += bn.reshape(num, 1, M)
    Y = act(Y)
    Y = (thetan.reshape(num, 1, 1, M) @ Y.reshape(num, T, M, 1)).reshape(num, T)
    return (X, Y)


def add_noise(data, noise=0.1):
    """
    :param noise: standard deviation of additive Gaussian noise
    """
    return data + np.random.normal(0, noise, data.shape)


def kl_divergence(y, y_hat, sigma):
    return np.mean((y - y_hat) ** 2, axis=-1) / (2 * sigma ** 2)
