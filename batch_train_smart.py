""" Given a configuration file, generates teacher networks, fits the data, and outputs the fitting results.
Automatic width tuning is performed.

The configuration file is a tsv with the following columns:
- d: input dimension
- M: width of teacher network
- noise: standard deviation of the Gaussian noise
- act: activation function of teacher network, must be one of 'relu' or 'sign'
- N: number of samples to generate for each teacher network
- hidden-layers: number of hidden layers in the fitting network
- patience: the patience for early stopping
- patience-tol: the relative tolerance for early stopping; if the best validation loss does not decrease by patience-tol in patience epochs, training is stopped.
- batch-size: the batch size used in training
- lr: learning rate; set to 0 for automatic learning rate tuning
- reduce-lr: set to 'T' to reduce learning rate on plateau
- dropout: set to 0 for the experiments in the paper
- weight-decay: set to 0 for the experiments in the paper
- count: number of experiments to run for the given configuration
- epochs: hard cap on number of epochs to train; set to 1500
- comment: any comment

"""

import smart_train
from fitting import *
import generate as generate
import logging
import time
import pandas as pd
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(SCRIPT_DIR) not in sys.path:
    sys.path.append(os.path.dirname(SCRIPT_DIR))


def main():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    )
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation Ratio")
    parser.add_argument(
        "--N-test", type=int, default=102400, help="Number of Test Samples")
    parser.add_argument(
        "--file", type=str, required=True, help="Config File Storing d, M, T, noise, hidden-dim, count, lr, weight-decay, dropout, patience, epochs, batch-size")

    args = parser.parse_args()

    train_file(args)


def get_hidden_dim_max(hidden_layers, M, d, T):
    """ Returns the maximum width of hidden layer to search for given the number of hidden layers and data generation parameters
    """
    if hidden_layers == 1:
        return 32 + 8 * np.minimum(
            np.sqrt(M * d) + np.maximum(M, d),
            T,
        )
    if hidden_layers == 2:
        return 32 + 2 * np.minimum(
            2 * np.sqrt(M * d) + 2 * np.maximum(M, d),
            2 * np.sqrt(T),
        )
    if hidden_layers == 3:
        return 16 + np.minimum(
            2 * np.sqrt(M * d) + 2 * np.maximum(M, d),
            2 * np.sqrt(T),
        )
    assert(False)


def train_file(args):
    config = pd.read_csv(
        args.file, sep="\t",
    )
    isSparse = 'K' in config.columns
    for _, row in config.iterrows():
        for _ in range(row["count"]):
            if isSparse:
                (thetan, an, bn) = generate.generate_single_sparse_layer(
                    row["K"], row["M"], row["d"], 1)
            else:
                (thetan, an, bn) = generate.generate_single_layer(
                    row["M"], row["d"], 1)
            (X, Y_noiseless) = generate.generate_single_data(
                row["N"], an, bn, thetan, row["act"])
            Y = generate.add_noise(Y_noiseless, row["noise"])
            hidden_dim_max = get_hidden_dim_max(
                row["hidden-layers"],
                row["M"],
                row["d"],
                row["N"],
            )
            start_time = time.time()
            logging.info("Start")
            logging.info(
                f"{row['d']}\t{row['M']}\t{row['N']}\t{row['noise']}"
                f"\t{row['act']}\t{row['dropout']}\t{row['hidden-layers']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['patience-tol']}\t{row['epochs']}"
                f"\t{row['reduce-lr']}"
                + (f"\t{row['K']}" if isSparse else ""),
            )
            hyperParamOpt = smart_train.hyper_param_search(
                X[0], Y[0],
                hidden_layers=row["hidden-layers"],
                hidden_dim_interval=(2, hidden_dim_max),
                val_ratio=args.val_ratio,
                lr=row["lr"],
                weight_decay=row["weight-decay"],
                dropout=row["dropout"],
                batch_size=row["batch-size"],
                patience=row["patience"],
                patience_tol=row["patience-tol"],
                epochs=row["epochs"],
                reduceLROnPlateau=None if row["reduce-lr"] != 'T' else True,
                verbose=False,
            )
            model = hyperParamOpt.model
            model.eval()
            (X_test, Y_test) = generate.generate_single_data(
                args.N_test, an, bn, thetan, row["act"])
            predicted = model(torch.Tensor(X_test)).detach().numpy()
            kl_divergence = generate.kl_divergence(
                Y_test.reshape(-1), predicted.reshape(-1), row["noise"])
            print(
                f"{row['d']}\t{row['M']}\t{row['N']}\t{row['noise']}\t{row['act']}"
                f"\t{kl_divergence}\t{row['hidden-layers']}"
                f"\t{hyperParamOpt.best_hidden_dim}\t{row['dropout']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['patience-tol']}\t{row['epochs']}"
                f"\t{hyperParamOpt.epoch_number}\t{hyperParamOpt.num_queries}"
                f"\t{row['reduce-lr']}"
                + (f"\t{row['K']}" if isSparse else ""),
                flush=True)
            end_time = time.time()
            logging.info(f"Finish: {end_time - start_time}")


if __name__ == "__main__":
    main()
