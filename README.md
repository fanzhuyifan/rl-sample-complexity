# About
# Overview of files
## Python source files
### `generate.py`
Helper functions related to generating single-hidden-layer teacher networks, generating data from these teacher networks, and calculating error.
The custom `FastTensorDataLoader` also saves the total number of queries to datapoints.
### `fitting.py`
Helper functions and classes for fitting teacher networks.
The main entrypoint is `train_one_model`.

### `find_lr.py`
Contains helper methods for automatically finding the learning rate (`find_lr`).

### `lowess.py`
Helper function for lowess regression and confidence interval.

### `smart_train.py`
Helper functions for automatic width selection using golden-section search.

### `batch_train.py`
The main module for running experiments *without* automatic width tuning.

### `batch_train_smart.py`
The main module for running experiments *with* automatic width tuning.

## Data Analysis Notebooks

## Other files

# Example Code