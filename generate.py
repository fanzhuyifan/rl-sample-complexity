""" Generate random single-hidden-layer teacher networks.
"""

import numpy as np


def generate_single_layer(
    M, d, num,
):
    """
    :param M: width (number of hidden neurons)
    :param d: input dimension
    :param num: number of teacher networks to generate
    :return: (thetan, an, bn), where thetan has shape (num, M), an has shape (num, M, d), and bn has shape (num, M)
    """
    an = np.random.normal(0, 1 / np.sqrt(d + 1), size=(num, M, d))
    bn = np.random.normal(0, 1 / np.sqrt(d + 1), size=(num, M))
    thetan = np.random.normal(0, 1 / np.sqrt(M), size=(num, M))
    return (thetan, an, bn)


def generate_single_sparse_layer(
    K, M, d, num,
):
    """
    :param K: sparsity parameter
    :param M: width (number of hidden neurons)
    :param d: input dimension
    :param num: number of teacher networks to generate
    :return: (thetan, an, bn), where thetan has shape (num, M), an has shape (num, M, d), and bn has shape (num, M)
    """
    an = np.random.normal(0, 1, size=(num, M, d))
    an /= np.linalg.norm(an, axis=-1, ord=2, keepdims=True)
    bn = np.zeros((num, M))
    alpha = np.ones(M) * K / M
    thetan = np.random.dirichlet(alpha, num)
    return (thetan, an, bn)


def generate_single_data(
    N, an, bn, thetan, act="sign",
):
    """ Generate noiseless data of length N according to the single-hidden-layer teacher network g specified by the parameters
    :param an: has shape (num, M, d)
    :param bn: has shape (num, M)
    :param thetan: shape (num, M)
    :param N: number of data to generate for each g
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
    X = np.random.normal(0, 1, (num, N, d))
    Y = (an.reshape(num, 1, M, 1, d) @
         X.reshape(num, N, 1, d, 1)).reshape(num, N, M)
    Y += bn.reshape(num, 1, M)
    Y = act(Y)
    Y = (thetan.reshape(num, 1, 1, M) @ Y.reshape(num, N, M, 1)).reshape(num, N)
    return (X, Y)


def add_noise(data, noise=0.1):
    """
    :param noise: standard deviation of additive Gaussian noise
    """
    return data + np.random.normal(0, noise, data.shape)


def kl_divergence(y, y_hat, sigma):
    """ Returns KL divergence of two normal distributions with standard deviation sigma.
    This is equal to L2 loss inversely scaled by noise.
    """
    return np.mean((y - y_hat) ** 2, axis=-1) / (2 * sigma ** 2)
