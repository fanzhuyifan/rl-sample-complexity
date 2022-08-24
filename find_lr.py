from copy import deepcopy
from matplotlib import pyplot as plt
import torch
import numpy as np


def combined(lrs: list, losses: list):
    """ Suggests a learning rate by combining the min_loss, valley, and steep methods.
    Uses the median of the result returned by the three methods
    """
    lr1, _ = min_loss(lrs, losses)
    lr2, _ = valley(lrs, losses)
    lr3, _ = steep(lrs, losses)
    lr = np.median([lr1, lr2, lr3])
    lrs_np = np.array(lrs)
    idx_new = np.searchsorted(lrs_np, lr)
    n = len(lrs)
    idx_new = max(0, min(n-1, idx_new))

    return lr, (lrs[idx_new], losses[idx_new])


def steep(lrs: list, losses: list, n_grids: int = 10):
    """ Suggests a learning rate by finding the index where descent is the sharpest

    https://gist.github.com/muellerzr/0634c486bd4939049314632ccc3f7a03?permalink_comment_id=3733835
    """
    n = len(lrs)
    grid_size = n // n_grids
    losses_np = np.array(losses)
    losses_grad = losses_np[grid_size:] - losses_np[:-grid_size]
    idx = np.argmin(losses_grad) + grid_size // 2
    return lrs[idx], (lrs[idx], losses[idx])


def min_loss(lrs: list, losses: list, ratio=1/20):
    """ Suggests a learning rate ratio of the minimum of the loss

    https://gist.github.com/muellerzr/0634c486bd4939049314632ccc3f7a03?permalink_comment_id=3733835
    """
    losses_np = np.array(losses)
    lrs_np = np.array(lrs)
    n = len(lrs)
    idx = np.argmin(losses_np)
    lr = lrs[idx] * ratio
    idx_new = np.searchsorted(lrs_np, lr)
    idx_new = max(0, min(n-1, idx_new))

    return lrs[idx], (lrs[idx], losses[idx])


def valley(lrs: list, losses: list, ratio=0.5):
    """Suggests a learning rate from the longest valley and returns its index

    https://gist.github.com/muellerzr/0634c486bd4939049314632ccc3f7a03?permalink_comment_id=3733835
    """
    n = len(losses)

    max_start, max_end = 0, 0

    # find the longest valley
    lds = [1]*n

    for i in range(1, n):
        for j in range(0, i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    idx = max_start + int((max_end - max_start) * ratio)

    return lrs[idx], (lrs[idx], losses[idx])


def find_lr(
    model,
    optimizer,
    loss_fn,
    data_loader,
    plot=False,
    min_lr=1e-8,
    max_lr=10,
    l=1.01,
    beta=0.98,
    method=combined,
):
    """ Find the learning rate by starting with a very small learning rate and exponentially increasing it each batch.
    Saves the states of the model and the optimizer and resets them before returning.

    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    :param model: the state of the model will be saved and reloaded
    :param optimizer: the optimizer to train the model
    :param plot: whether to display a plot of loss vs lr
    :param min_lr: minimum learning rate
    :param max_lr: maximum learning rate
    :param l: exponential factor for increasing the learning rate each batch
    :param beta: parameter for exponentially smoothing the losses
    :param method: method for finding the best learning rate
    :return: optimum learning rate
    """
    model_state = deepcopy(model.state_dict())
    opt_state = deepcopy(optimizer.state_dict())

    lrs, losses = _find_lr_train(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        data_loader=data_loader, min_lr=min_lr, max_lr=max_lr,
        l=l, beta=beta,
    )
    if type(method) is list:
        lr = []
        point = []
        for m in method:
            a, b = m(lrs, losses)
            lr.append(a)
            point.append(b)
    else:
        lr, point = method(lrs, losses)
    if plot:
        ax = plt.subplot()
        ax.plot(
            lrs,
            losses,
        )
        if type(method) is list:
            for p in point:
                ax.plot(*p, 'o')
            ax.legend(['loss vs lr'] + [m.__name__ for m in method])
        else:
            ax.plot(*point, 'o')
            ax.legend(['loss vs lr', method.__name__])
        ax.set_xscale('log')
        ax.set_xlabel('lr')
        ax.set_ylabel('loss')
        plt.show()

    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    return lr


def _find_lr_train(
    model,
    optimizer,
    loss_fn,
    data_loader,
    min_lr,
    max_lr,
    l,
    beta,
):
    """ Perform training for find_lr.
    Stops training if smoothed_loss > 4 * min_loss
    """
    for g in optimizer.param_groups:
        g['lr'] = min_lr

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda _: l,
    )
    losses = []
    lrs = []

    min_loss = np.inf
    avg_loss = 0
    batch_num = 0
    while scheduler.get_last_lr()[0] < max_lr:
        for _, data in enumerate(data_loader):
            batch_num += 1
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad(set_to_none=True)

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            losses.append(smoothed_loss)
            lrs.append(scheduler.get_last_lr()[0])
            min_loss = min(min_loss, smoothed_loss)
            if smoothed_loss > 4 * min_loss:
                return lrs, losses

            scheduler.step()
            if scheduler.get_last_lr()[0] > max_lr:
                return lrs, losses
    return lrs, losses
