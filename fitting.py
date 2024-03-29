""" Contains classes and functions for fitting teacher networks.
"""

import logging
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from find_lr import find_lr


class SingleLayer(nn.Module):
    """ Single Layer Model
    """

    def __init__(self, d, hidden_size, act=nn.ReLU()):
        super(SingleLayer, self).__init__()
        self.fc1 = nn.Linear(d, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act = act

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiLayer(nn.Module):
    """ Multi Layer Model
    """

    def __init__(self, d, hidden_size, act=nn.ReLU(), dropout=0, batchNorm=False):
        super(MultiLayer, self).__init__()
        self.n_hidden = len(hidden_size)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, hidden_size[0]))
        for i in range(self.n_hidden - 1):
            self.fcs.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.fcs.append(nn.Linear(hidden_size[-1], 1))
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.batchNorm = batchNorm
        if self.batchNorm:
            self.batch_norms = nn.ModuleList()
            for hidden_size in hidden_size:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        x = self.fcs[0](x)
        for i in range(self.n_hidden):
            x = self.dropout(x)
            if self.batchNorm:
                x = self.batch_norms[i](x)
            x = self.act(x)
            x = self.fcs[i+1](x)
        return x


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Saves the total number of queries to datapoints.

    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/4
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_queries = 0

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices)
                          for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size]
                          for t in self.tensors)
        self.i += self.batch_size
        self.num_queries += batch[0].shape[0]
        return batch

    def __len__(self):
        return self.n_batches


def get_data_loader(x, y, batch_size=100):
    """ Returns data_loader from numpy arrays.
    """
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y[..., np.newaxis])
    dataloader = FastTensorDataLoader(
        tensor_x, tensor_y, batch_size=batch_size)
    return dataloader


def train_one_epoch(
    model,
    optimizer,
    loss_fn,
    training_loader,
    schedular=None,
):
    """ Trains model for one epoch
    :return: average training loss
    """
    running_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
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
        running_loss += loss.item()

        if schedular is not None:
            schedular.step()

    return running_loss / (i+1)


def train_early_stopping(
    model,
    optimizer,
    loss_fn,
    training_loader,
    validation_loader,
    epochs=1000,
    patience=5,
    patience_tol=0,
    cyclicLR=None,
    reduceLROnPlateau=None,
    verbose=False,
):
    """ Train model with early stopping
    :param patience: Number of epochs with no improvement after which training will be stopped
    :param patience_tol: the relative tolerance for loss in early stopping; training will stop if loss do not decrease by patience_tol (relatively) in patience epochs.
    :return: best model (on validation set)
    """
    from datetime import datetime
    # import time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if verbose:
        writer = SummaryWriter('runs/{}'.format(timestamp))
    best_vlosses = [np.inf]
    last_lr = None

    for epoch in range(epochs):
        if verbose:
            print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            model, optimizer, loss_fn, training_loader, cyclicLR)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss.detach().numpy() / (i + 1)

        if reduceLROnPlateau is not None:
            reduceLROnPlateau.step(avg_vloss)
            cur_lr = optimizer.param_groups[0]['lr']
            if last_lr != cur_lr:
                logging.info(
                    f"lr update: {cur_lr} epoch {epoch} vloss {avg_vloss}")
            last_lr = cur_lr

        if verbose:
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            writer.add_scalars(
                'LOSS',
                {'Training': avg_loss, 'Validation': avg_vloss},
                epoch + 1
            )
            writer.add_scalar(
                'LR',
                optimizer.param_groups[0]['lr'],
                epoch + 1
            )
            writer.flush()

        if avg_vloss < best_vlosses[-1]:
            best_model_state = deepcopy(model.state_dict())
            best_avg_loss = avg_loss
            best_vlosses.append(avg_vloss)
        else:
            best_vlosses.append(best_vlosses[-1])
        if len(best_vlosses) > patience and best_vlosses[-1] >= best_vlosses[-patience-1] * (1 - patience_tol):
            break

    model.load_state_dict(best_model_state)

    return (epoch + 1, best_vlosses[-1], best_avg_loss)


def train_one_model(
    hidden_dim, x, y,
    val_ratio=0.2,
    lr=0,
    weight_decay=0,
    dropout=0,
    batch_size=64,
    val_batch_size=4096,
    model=None,
    act=nn.ReLU(),
    batchNorm=False,
    cyclicLR=None,
    reduceLROnPlateau=None,
    **train_kwargs,
):
    """ Sets up training and trains one model.
    :param lr: learning rate; if set to 0, will automatically choose learning rate via find_lr
    """
    (T, d) = x.shape
    val_size = int(T * val_ratio + 1)
    if model is None:
        model = MultiLayer(d, hidden_dim, dropout=dropout,
                           act=act, batchNorm=batchNorm)
    training_loader = get_data_loader(x[:-val_size], y[:-val_size], batch_size)
    validation_loader = get_data_loader(
        x[-val_size:], y[-val_size:], val_batch_size)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    if cyclicLR:
        cyclicLR = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=lr / 10, max_lr=lr,
            step_size_up=3 * ((T + batch_size - 1) // batch_size),
            cycle_momentum=False,
        )
    if reduceLROnPlateau:
        reduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1,
            patience=12,
        )
    if lr == 0:
        lr = find_lr(
            model=model, optimizer=optimizer, loss_fn=loss_fn,
            data_loader=training_loader, plot=False,
        )
        for g in optimizer.param_groups:
            g['lr'] = lr
        logging.info(f"automatically setting lr to {lr}")
    (epoch_number, best_vloss, avg_loss) = train_early_stopping(
        model,
        optimizer,
        loss_fn,
        training_loader,
        validation_loader,
        cyclicLR=cyclicLR,
        reduceLROnPlateau=reduceLROnPlateau,
        ** train_kwargs,
    )
    return (
        model, epoch_number, best_vloss, avg_loss,
        training_loader.num_queries + validation_loader.num_queries,
    )
