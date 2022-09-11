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


def get_N_e_max(data, epsilon):
    return data.where(data['epsilon'] < epsilon).groupby(['d', 'M', 'noise']).agg(
        N_e_max=pd.NamedAgg(column="N", aggfunc="min"),
    ).reset_index()


def get_N_e_min(data, epsilon):
    return data.where(data['epsilon'] >= epsilon).groupby(['d', 'M', 'noise']).agg(
        N_e_min=pd.NamedAgg(column="N", aggfunc="max"),
    ).reset_index()


def get_next_tests(data, epsilon, candidates):
    data_min_max = get_N_e_max(data, epsilon).merge(
        get_N_e_min(data, epsilon),
        on=['d', 'M', 'noise'],
        how='outer',
    )

    temp1 = data_min_max[['d', 'M', 'noise', 'N_e_max']].dropna()
    temp1['N'] = (temp1['N_e_max'] / 2).astype('int')
    temp1 = temp1.drop(['N_e_max'], axis=1)

    temp2 = data_min_max[['d', 'M', 'noise', 'N_e_min']].dropna()
    temp2['N'] = (temp2['N_e_min'] * 2).astype('int')
    temp2 = temp2.drop(['N_e_min'], axis=1)

    temp3 = data_min_max.dropna()[['d', 'M', 'noise', 'N_e_max']]
    temp3['N'] = (temp3['N_e_max']).astype('int')
    temp3 = temp3.drop(['N_e_max'], axis=1)
    temp3['d'] *= 2

    temp4 = data_min_max.dropna()[['d', 'M', 'noise', 'N_e_max']]
    temp4['N'] = (temp4['N_e_max']).astype('int')
    temp4 = temp4.drop(['N_e_max'], axis=1)
    temp4['M'] *= 2

    explore = pd.concat([temp3, temp4]).groupby(['d', 'M', 'noise']).agg(
        N=pd.NamedAgg(column="N", aggfunc="max"),
    ).reset_index()
    candidates_with_data = data_min_max[
        (~data_min_max['N_e_max'].isna())
        |
        (~data_min_max['N_e_min'].isna())
    ][['M', 'd', 'noise']]
    candidates_without_data = pd.concat([
        candidates_with_data, candidates_with_data, candidates
    ]).drop_duplicates(keep=False)
    explore = candidates_without_data.merge(
        explore,
        left_on=['M', 'd', 'noise'],
        right_on=['M', 'd', 'noise'],
        how='inner',
    )

    result = pd.concat([temp1, temp2, explore]).drop_duplicates()

    temp = data[['d', "M", 'noise', 'N']].drop_duplicates()
    result = pd.concat([temp, temp, result]).drop_duplicates(
        keep=False
    ).sort_values([
        'noise', 'd', 'M', 'N'
    ])

    result = candidates.merge(
        result,
        left_on=['d', 'M', 'noise'],
        right_on=['d', 'M', 'noise'],
        how='left',
    ).dropna().drop_duplicates()
    return result


def generate_target_epsilon(args):
    analysis = pd.read_csv(args.file)
    candidates = analysis[['d', 'M', 'noise']].drop_duplicates()
    result = get_next_tests(analysis, args.epsilon, candidates)
    data = add_cols(result, args)
    data.to_csv(args.output, sep='\t', index=False)


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
    parser_epsilon.add_argument(
        "-f",
        "--file",
        help="The reference *analysis* file containing previous results",
        type=str,
        required=True,
    )

    parser_fixed_N = subparsers.add_parser(
        'fixed-N',
        help="Generate config where the number of samples is fixed",
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
    elif args.subcommand == 'target-epsilon':
        generate_target_epsilon(args)
    print(args)


if __name__ == "__main__":
    main()
