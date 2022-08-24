import torch.nn as nn
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
        "--file", type=str, required=True, help="Config File Storing d, M, T, noise, hidden-dim, count, lr, weight-decay, dropout, patience, epochs, batch-size")

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
                f"{row['d']}\t{row['M']}\t{row["N"]}\t{row['noise']}"
                f"\t{row['act']}\t{row['dropout']}"
                f"\t{row['model']}\t{row['hidden-layers']}\t{row['trials']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['patience-tol']}\t{row['epochs']}"
                f"\t{row['reduce-lr']}",
            )
            best_model = None
            best_vloss = np.inf
            total_num_queries = 0
            for _ in range(row["trials"]):
                (model, epoch_number, vloss, train_loss, num_queries) = train_one_model(
                    [row["model"]] * row["hidden-layers"], X[0], Y[0],
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
                f"{row['d']}\t{row['M']}\t{row["N"]}\t{row['noise']}\t{row['act']}"
                f"\t{kl_divergence}\t{row['hidden-layers']}"
                f"\t{row['model']}\t{row['trials']}\t{row['dropout']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['patience-tol']}\t{row['epochs']}"
                f"\t{epoch_number}\t{total_num_queries}"
                f"\t{row['reduce-lr']}",
                flush=True)
            end_time = time.time()
            logging.info(f"Finish: {end_time - start_time}")


if __name__ == "__main__":
    main()
