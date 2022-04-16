""" Train a model with hyperparameter search
"""

import scipy as sp
import numpy as np
from scipy import optimize
from single_layer import train_one_model


class HyperParamOpt:
    """ Class for optimizing hyperparameters.
    Keeps track of the best performing model and the corresponding hyperparameters
    """

    def __init__(self, X, Y, log_hidden_dim_interval, **fixed_params) -> None:
        self.log_hidden_dim_interval = log_hidden_dim_interval
        self.fixed_params = fixed_params
        self.X = X
        self.Y = Y

        self.best_hidden_dim = None
        self.model = None
        self.best_loss = np.inf
        self.epoch_number = None

    def train(self, log_hidden_dim):
        if (log_hidden_dim < self.log_hidden_dim_interval[0]
                or log_hidden_dim > self.log_hidden_dim_interval[1]):
            return np.inf

        hidden_dim = int(np.power(2, log_hidden_dim))
        (model, epoch_number, best_vloss, train_loss) = train_one_model(
            [hidden_dim, hidden_dim, hidden_dim], self.X, self.Y,
            **self.fixed_params,
        )
        best_vloss = best_vloss.detach().numpy()
        if best_vloss < self.best_loss:
            self.best_loss = best_vloss
            self.best_hidden_dim = hidden_dim
            self.model = model
            self.epoch_number = epoch_number
        return best_vloss

    def __str__(self) -> str:
        return (
            f"best_loss: {self.best_loss}, "
            f"model: {self.model}, best_hidden_dim: {self.best_hidden_dim}, "
            f"epoch_number: {self.epoch_number}, "
            f"fixed_params: {self.fixed_params}"
        )


def hyper_param_search(
    X,
    Y,
    hidden_dim_interval,
    maxiter=100,
    **hyper_params
):
    """ Search in the space of hidden dimension sizes fixing other hyperparameters, and assuming the validation loss is a unimodal function of hidden dimension size.
    """
    log_hidden_dim_interval = (
        np.log2(hidden_dim_interval[0]), np.log2(hidden_dim_interval[1]))
    hyperParamOpt = HyperParamOpt(
        X, Y,
        log_hidden_dim_interval,
        **hyper_params
    )
    result = optimize.golden(
        hyperParamOpt.train,
        brack=log_hidden_dim_interval,
        tol=0.25,
        maxiter=maxiter,
        full_output=True,
    )
    return hyperParamOpt
