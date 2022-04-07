import numpy as np
import pandas as pd
from tqdm import tqdm
import generate as generate

def calc_init_regret(d, M, noise, N_test, N_trials):
    result = np.zeros(N_trials)
    for i in range(N_trials):
        (thetan, an, bn) = generate.generate_single_layer_v2(M, d, 1)
        (_, Y_test) = generate.generate_single_data_v2(N_test, an, bn, thetan)
        result[i] = generate.kl_divergence(Y_test.reshape(-1), 0, noise)
    return result

def main(filename, N_test, N_trials):
    data = pd.read_excel(filename)
    (N, n_cols) = data.shape
    assert(n_cols == 5)
    for i in tqdm(range(N)):
        regrets = calc_init_regret(data['d'][i], data['M'][i], data['noise'][i], N_test, N_trials)
        data.at[i, 'initial-regret'] = np.mean(regrets)
        # standard deviation of sample mean needs to be divided by sqrt(N_trials)
        data.at[i, 'std-dev'] = np.std(regrets) / np.sqrt(N_trials)
    print(data)
    data.to_excel(filename, index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python calc_init_regret.py filename N_test N_trials")
    else:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))