# About
# Overview of files
## `generate.py`
Helper functions related to generating single-hidden-layer teacher networks, generating data from these teacher networks, and calculating error.
The custom `FastTensorDataLoader` also saves the total number of queries to datapoints.
## `fitting.py`
Helper functions and classes for fitting teacher networks.
The main entrypoint is `train_one_model`.

## `find_lr.py`
Contains helper methods for automatically finding the learning rate (`find_lr`).

# Example Code