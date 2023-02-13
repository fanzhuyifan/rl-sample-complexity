import pandas as pd
import numpy as np
import os


def summarize(input_file, output_file):
    if os.path.splitext(input_file)[1] in [".csv", ".tsv"]:
        data = pd.read_table(input_file)
    else:
        data = pd.read_excel(input_file)
    if data.duplicated().any():
        print("Contains duplicates!")
        print(data.duplicated().sum())
    else:
        print("No Duplicates!")
    groupby_cols = [
        'd', 'M', 'N', 'act', 'noise', 'hidden-layers',
        'patience', 'patience-tol', 'epochs',
        'lr', 'reduce-lr'
    ]
    if "K" in data.columns:
        groupby_cols.append("K")
    if "trials" in data.columns:
        groupby_cols.append("trials")
    data = data.groupby(groupby_cols).agg(
        regret_mean=pd.NamedAgg(column="regret", aggfunc="mean"),
        regret_median=pd.NamedAgg(column="regret", aggfunc="median"),
        regret_std=pd.NamedAgg(column="regret", aggfunc="std"),
        count=pd.NamedAgg(column="regret", aggfunc="count"),
        num_queries_mean=pd.NamedAgg(column="num_queries", aggfunc="mean"),
        num_queries_std=pd.NamedAgg(column="num_queries", aggfunc="std"),
    ).reset_index()
    data["regret_std"] /= np.sqrt(data["count"])
    data["num_queries_std"] /= np.sqrt(data["count"])
    data["epsilon"] = data["regret_mean"]  # / data["initial-regret"]
    data["epsilon_std"] = data["regret_std"]  # / data["initial-regret"]
    data = data.sort_values(
        by=['noise', 'd', 'M', 'N', 'hidden-layers'], ascending=True)
    if os.path.splitext(output_file)[1] in [".csv", ".tsv"]:
        data.to_csv(output_file, index=False)
    else:
        data.to_excel(output_file, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input File",
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output File",
        required=True
    )

    args = parser.parse_args()
    summarize(args.input, args.output)
    pass


if __name__ == "__main__":
    main()
