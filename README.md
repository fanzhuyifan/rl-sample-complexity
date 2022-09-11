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

### `generate_config.py`
Helper file for generating tsv configs for `batch_train.py` and `batch_train_smart.py`.

## Other files
### `result_header_tune.tsv`
Header for tsv file for results with width tuning.

### `result_header_no_tune.tsv`
Header for tsv file for results with width tuning.

# Example Code
```bash
python generate_config.py -w tune --hidden 1 --count 2 -o config_1.tsv fixed-N -N 256 -d 16 -M 16 -n 0.1 0.2
```
Generates a config file where
- the width of the fitting network is automatically tuned
- the number of hidden layers in the fitting network is 1
- 2 experiments are run for each configuration
- the sample size is fixed at 256
- d lies in (1,2,4,8,16)
- M lies in (1,2,4,8,16)
- sigma lies in (0.1,0.2)

```bash
cp result_header_tune.tsv result_1.tsv && python batch_train_smart.py --file config_1.tsv 2>/dev/null >> result_1.tsv
```
Trains on the previous configuration file, discarding debug output and storing the result into `result_1.tsv`.

```bash
python summarize_results.py -i result_1.tsv -o analysis_1.csv
```
Summarizes the results and stores into `analysis_1.csv`.

---------- 

```bash
python generate_config.py -w 4M --hidden 1 --count 2 -o config_2.tsv fixed-N -N 256 -d 16 -M 16 -n 0.1 0.2
```
Generates a config file where
- the width of the fitting network is 4 times the generating width
- the number of hidden layers in the fitting network is 1
- 2 experiments are run for each configuration
- the sample size is fixed at 256
- d lies in (1,2,4,8,16)
- M lies in (1,2,4,8,16)
- sigma lies in (0.1,0.2)

```bash
cp result_header_no_tune.tsv result_2.tsv && python batch_train.py --file config_2.tsv 2>/dev/null >> result_2.tsv
```
Trains on the previous configuration file, discarding debug output and storing the result into `result_2.tsv`.

```bash
python summarize_results.py -i result_2.tsv -o analysis_2.csv
```
Summarizes the results and stores into `analysis_2.csv`.

----- 

```bash
python3 generate_config.py -w tune --hidden 1 --count 2 -o config_3.tsv target-epsilon -e 0.1 -f analysis_1.csv
```
Based on the results in `analysis_1.csv`, try to reach a target epsilon of 0.1 for all (d, M, noise) tuples in `analysis_1.csv`.
The sample size N is doubled if epsilon is too big, and halved otherwise.

-----
```bash
python3 generate_config.py -w best --hidden 1 --count 2 -o config_4.tsv --reference result_1.tsv duplicate --config config_1.tsv
```
Generate a configuration file just like `config_1.tsv`, but with the width set to the best width.

# On parallel computation
Since the batch size is fixed to something small, we discovered that the speed does not improve with multi-threading.
Hence, to improve throughput, we executed different jobs on different cores using slurm.
Without a slurm cluster, this could be done with `gnu parallel`.