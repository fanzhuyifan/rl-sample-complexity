""" Given a configuration file, generates teacher networks, fits the data, and outputs the fitting results.
No automatic width tuning is performed.

The configuration file is a tsv with the following columns:
- d: input dimension
- M: width of teacher network
- noise: standard deviation of the Gaussian noise
- act: activation function of teacher network, must be one of 'relu' or 'sign'
- N: number of samples to generate for each teacher network
- hidden-layers: number of hidden layers in the fitting network
- fitting-width: the width of the fitting network
- patience: the patience for early stopping
- patience-tol: the relative tolerance for early stopping; if the best validation loss does not decrease by patience-tol in patience epochs, training is stopped.
- lr: learning rate; set to 0 for automatic learning rate tuning
- reduce-lr: set to 'T' to reduce learning rate on plateau
- dropout: set to 0 for the experiments in the paper
- weight-decay: set to 0 for the experiments in the paper
- trials: number of times to fit a teacher network; the best among different trials is used for final evaluation; always set to 1 for the experiments in the paper
- count: number of experiments to run for the given configuration
- epochs: hard cap on number of epochs to train; set to 1500
- comment: any comment

"""

import pandas as pd
import sys
import time
import logging
import generate as generate
from fitting import *


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
        "--file", type=str, required=True, help="Config File")

    args = parser.parse_args()

    train_file(args)


def train_file(args):
    config = pd.read_csv(
        args.file, sep="\t",
    )
    for _, row in config.iterrows():
        for _ in range(row["count"]):
            (thetan, an, bn) = generate.generate_single_layer(
                row["M"], row["d"], 1)
            (X, Y_noiseless) = generate.generate_single_data(
                row["N"], an, bn, thetan, row["act"])
            Y = generate.add_noise(Y_noiseless, row["noise"])
            start_time = time.time()
            logging.info("Start")
            logging.info(
                f"{row['d']}\t{row['M']}\t{row['N']}\t{row['noise']}"
                f"\t{row['act']}\t{row['dropout']}"
                f"\t{row['fitting-width']}\t{row['hidden-layers']}\t{row['trials']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['patience-tol']}\t{row['epochs']}"
                f"\t{row['reduce-lr']}",
            )
            best_model = None
            best_vloss = np.inf
            total_num_queries = 0
            for _ in range(row["trials"]):
                (model, epoch_number, vloss, train_loss, num_queries) = train_one_model(
                    [row["fitting-width"]] * row["hidden-layers"], X[0], Y[0],
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
                total_num_queries += num_queries
                if vloss < best_vloss:
                    best_vloss = vloss
                    best_model = model
                    best_model.eval()
                    (X_test, Y_test) = generate.generate_single_data(
                        args.N_test, an, bn, thetan, row["act"])
                    predicted = best_model(
                        torch.Tensor(X_test)).detach().numpy()
                    kl_divergence = generate.kl_divergence(
                        Y_test.reshape(-1), predicted.reshape(-1), row["noise"])
            best_model.eval()
            (X_test, Y_test) = generate.generate_single_data(
                args.N_test, an, bn, thetan, row["act"])
            predicted = best_model(torch.Tensor(X_test)).detach().numpy()
            kl_divergence = generate.kl_divergence(
                Y_test.reshape(-1), predicted.reshape(-1), row["noise"])
            print(
                f"{row['d']}\t{row['M']}\t{row['N']}\t{row['noise']}\t{row['act']}"
                f"\t{kl_divergence}\t{row['hidden-layers']}"
                f"\t{row['fitting-width']}\t{row['trials']}\t{row['dropout']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['patience-tol']}\t{row['epochs']}"
                f"\t{epoch_number}\t{total_num_queries}"
                f"\t{row['reduce-lr']}",
                flush=True)
            end_time = time.time()
            logging.info(f"Finish: {end_time - start_time}")


if __name__ == "__main__":
    main()
