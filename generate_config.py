""" This file helps users generate config files for training
"""

import numpy as np
import pandas as pd


def add_cols(data, args):
    data['act'] = "relu"
    data['patience'] = 40
    data['patience-tol'] = 0.01
    data['batch-size'] = 64
    data['epochs'] = 1500
    data['weight-decay'] = 0
    data['lr'] = 0
    data['dropout'] = 0
    data = data.drop_duplicates()
    data['comment'] = 'reduce-lr'
    data['reduce-lr'] = 'T'
    data['trials'] = 1
    data['count'] = args.count
    data['hidden-layers'] = args.hidden
    if args.width == '4M':
        data['fitting-width'] = 4 * data['M']
    elif args.width == 'same':
        data['fitting-width'] = data['M']

    return data


def get_candidates(maxLogD, maxLogM, noises):
    candidates = pd.DataFrame(data={
        'd': [2 ** i for i in range(maxLogD + 1)]
    }).merge(
        pd.DataFrame(data={
            'M': [2 ** i for i in range(maxLogM + 1)]
        }),
        how='cross',
    ).merge(
        pd.DataFrame(data={
            'noise': noises,
        }),
        how='cross',
    )
    return candidates


def generate_fixed_N(args):
    candidates = get_candidates(
        int(np.log2(args.d)),
        int(np.log2(args.M)),
        args.noises,
    )
    data = add_cols(candidates, args)
    data['N'] = args.N
    data.to_csv(args.output, sep='\t', index=False)
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-w",
        "--width",
        help="Width Tuning Method",
        choices=["tune", "same", "4M"],
        required=True
    )
    parser.add_argument(
        "--hidden",
        help="Number of hidden layers in fitting network",
        type=int,
        choices=[1, 2, 3],
        required=True
    )
    parser.add_argument(
        "--count",
        help="Number of experiments to run for each configuration",
        type=int,
        required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file name",
        required=True,
    )

    subparsers = parser.add_subparsers(
        description="Type of config generation to perform",
        dest='subcommand',
        required=True,
    )

    parser_epsilon = subparsers.add_parser(
        'target-epsilon',
        help="Try to reach a target epsilon (for each (d, M, noise) pair, make sure the biggest error is above epsilon and the smallest error is below epsilon, by doubling and halving the sample size)",
    )
    parser_epsilon.add_argument(
        "-e",
        "--epsilon",
        help="The target epsilon",
        type=float,
        required=True,
    )

    parser_fixed_N = subparsers.add_parser(
        'fixed-N',
        help="123",
    )
    parser_fixed_N.add_argument(
        "-N",
        help="Number of samples",
        required=True,
        type=int,
    )

    def check_power_2(value):
        ivalue = int(value)
        if (ivalue & (ivalue-1) == 0) and ivalue != 0:
            return ivalue
        raise argparse.ArgumentTypeError(f"{value} is not a power of 2")

    parser_fixed_N.add_argument(
        "-d",
        help="Maximum input dimension",
        required=True,
        type=check_power_2,
    )

    parser_fixed_N.add_argument(
        "-M",
        help="Maximum width of teacher network",
        required=True,
        type=check_power_2,
    )

    parser_fixed_N.add_argument(
        '-n', '--noises',
        nargs='+',
        help='Standard Deviation of the noises',
        required=True,
    )

    args = parser.parse_args()
    if args.subcommand == 'fixed-N':
        generate_fixed_N(args)
    print(args)


if __name__ == "__main__":
    main()
