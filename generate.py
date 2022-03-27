""" Generate random data-generating environments
"""

import numpy as np

def generate_activations(
    d, N,
):
    """ Generate activation units from R^d to R of the form 
    \phi(u)=sign(a^T u + b), 
    where a is in R^d and b is in R
    :param d: dimension of input
    :param N: number of activation units to generate
    :return: (an, bn). an has shape (N, d), and bn has shape (N).
    """
    an = np.random.normal(0, 1, size=(N, d))
    bn = np.random.normal(0, 1, size=(N))
    return (an, bn)

def generate_single_layer(
    N, M, d, num, an, bn
):
    """ Generates random function g from R^d to R.
    Each g is characterized by 
        M indices i_1, ..., i_M
        M thetas in R; theta_1,...,theta_M;
    such that g(u)=\sum_{j=1}^M\theta_j sign(a_{i_j}^T u + b_{i_j}),
    where the {a}s and {b}s are generated according to generate_activations

    :param N: number of activation units to use
    :param M: the sparsity
    :param d: the dimension of the input
    :param num: number of gs to generate
    :param an: shape (N, d)
    :param bn: shape (N)
    :return: (In, thetan), where In has shape (num, M), and thetan has shape (num, M)
    """
    In = np.array([np.random.choice(N, M, replace=False) for _ in range(num)])
    thetan = np.random.normal(0, np.sqrt(1 / M), (num, M))
    return (In, thetan)

def generate_single_data(
    T, an, bn, In, thetan,
):
    """ Generate data of length T according to the single layer generation process g specified by the parameters
    :param an: has shape (N, d)
    :param bn: has shape (N)
    :param In: shape (num, M)
    :param thetan: shape (num, M)
    :param T: number of data to generate for each g
    :return: (X, Y), where X has shape (num, T, d), and Y has shape (num, T).
    X has standard Gaussian distribution, and the corresponding Y=g(X)
    """
    (N, d) = an.shape
    (num, M) = In.shape
    # X: (num, T, d)
    X = np.random.normal(0, 1, (num, T, d))
    # an[In, :]: (num, M, d)
    # temp: (num, T, M); a^T X
    temp = (an[In, :].reshape(num, 1, M, 1, d) @ X.reshape(num, T, 1, d, 1)).reshape(num, T, M)
    # bn[In]: (num, M)
    # acts: (num, T, M)
    acts = np.sign(temp + bn[In].reshape(num, 1, M))
    Y = (thetan.reshape(num, 1, 1, M) @ acts.reshape(num, T, M, 1)).reshape(num, T)
    return (X, Y)