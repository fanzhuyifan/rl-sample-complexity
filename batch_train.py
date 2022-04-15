import generate as generate
from single_layer import *
import torch.nn as nn
import pandas as pd

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", type=int, default=8, help="dimension of input")
    parser.add_argument(
        "-M", type=int, default=2, help="Sparsity")
    parser.add_argument(
        "-T", nargs="+", type=int, default=[1024], help="Number of Samples")
    parser.add_argument(
        "--N-test", type=int, default=102400, help="Number of Test Samples")
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Standard Deviation of Noise")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning Rate")
    parser.add_argument(
        "--weight-decay", type=float, default=0.001, help="Weight Decay")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument(
        "--num-experiments", type=int, default=64, help="Number of experiments to perform")
    parser.add_argument(
        "--hidden-dim", type=int, nargs="+", default=[64, 64, 64], help="Number of unts in hidden dimension")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation Ratio")
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch Size")
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience in early stopping")
    parser.add_argument(
        "--epochs", type=int, default=500, help="Maximum number of epochs")
    parser.add_argument(
        "--file", type=str, default=None, help="Config File Storing d, M, T, noise, hidden-dim, count, lr, weight-decay, dropout, patience, epochs, batch-size")

    args = parser.parse_args()
    if args.file is None:
        train(args)
    else: 
        train_file(args)

def train_file(args):
    config = pd.read_csv(args.file, sep="\t")
    for _, row in config.iterrows():
        hidden_dim = eval(row["model"])
        for _ in range(row["count"]):
            (thetan, an, bn) = generate.generate_single_layer_v2(row["M"], row["d"], 1)
            (X, Y_noiseless) = generate.generate_single_data_v2(row["T"], an, bn, thetan)
            Y = generate.add_noise(Y_noiseless, row["noise"])
            input = X[0]
            (model, epoch_number, best_vloss, train_loss) = train_one_model(
                hidden_dim, X[0], Y[0], 
                val_ratio=args.val_ratio, 
                lr=row["lr"], 
                weight_decay=row["weight-decay"],
                dropout=row["dropout"],
                batch_size=row["batch-size"],
                patience=row["patience"], 
                epochs=row["epochs"],
                verbose=False,
            )
            model.eval()
            (X_test, Y_test) = generate.generate_single_data_v2(args.N_test, an, bn, thetan)
            predicted = model(torch.Tensor(X_test)).detach().numpy()
            kl_divergence = generate.kl_divergence(Y_test, predicted.reshape(-1), args.noise)
            print(
                f"{row['d']}\t{row['M']}\t{row['T']}\t{row['noise']}"
                f"\t{kl_divergence[0]}\t{hidden_dim}\t{row['dropout']}"
                f"\t{row['weight-decay']}\t{row['lr']}\t{row['batch-size']}"
                f"\t{row['patience']}\t{row['epochs']}\t{epoch_number}",
                flush=True)


def train(args):
    for T in args.T:
        for _ in range(args.num_experiments):
            (thetan, an, bn) = generate.generate_single_layer_v2(args.M, args.d, 1)
            (X, Y_noiseless) = generate.generate_single_data_v2(T, an, bn, thetan)
            Y = generate.add_noise(Y_noiseless, args.noise)
            input = X[0]
            (model, epoch_number, best_vloss, train_loss) = train_one_model(
                args.hidden_dim, X[0], Y[0], 
                val_ratio=args.val_ratio, 
                lr=args.lr, 
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                batch_size=args.batch_size,
                patience=args.patience, 
                epochs=args.epochs,
                verbose=False,
            )
            model.eval()
            (X_test, Y_test) = generate.generate_single_data_v2(args.N_test, an, bn, thetan)
            predicted = model(torch.Tensor(X_test)).detach().numpy()
            kl_divergence = generate.kl_divergence(Y_test, predicted.reshape(-1), args.noise)
            print(f"{args.d}\t{args.M}\t{T}\t{args.noise}\t{kl_divergence[0]}\t{args.hidden_dim}\t{args.dropout}\t{args.weight_decay}", flush=True)

if __name__ == "__main__":
    main()