# Duhamel Neural Operator

Code for **"Duhamel Neural Operator for Learning Phase-Field Dynamics"**

Duhamel-NO advances a phase field one frame at a time. A **Fourier SSM**, measured from simulations, moves every Fourier
mode by its exact linear factor, and a **U-shaped neural operator** learns the nonlinear correction.

## Quick start: reproduce the paper's numbers (5 minutes, no GPU, no download)

```bash
# download the zip from https://anonymous.4open.science/r/Duhamel-Neural-Operator-8DF4/
# (click "Download Repository"), then:
unzip Duhamel-Neural-Operator-8DF4.zip -d Duhamel-Neural-Operator
cd Duhamel-Neural-Operator
pip install numpy scipy matplotlib       # this step needs nothing else; requirements.txt has the full environment
bash scripts/make_tables.sh
```

This rebuilds every number (77) and every table (10) of the paper, main text and appendix, from the included result
files, and compares them with the values printed in the paper. The last line should be `ALL MATCH`.

## Data and trained models

| what | where |
|---|---|
| D1, D2 and hysteresis data, with the paper's splits | [anonymous link](https://anonymous-hf.com/a/58mhxpc51v6u/) (dataset) |
| every trained model of the paper (109 checkpoints) and the measured Fourier SSMs | [anonymous link](https://anonymous-hf.com/a/9z48lcxh6uv0/) (models) |

The anonymous mirrors have no `hf download`, and their file browser shows at most 1000 files per folder (D1 has 1001),
so download with `scripts/download.py` (standard library only). It verifies every file's SHA-256 and resumes after an
interruption:

```bash
python scripts/download.py data --out ./ferro_data                       # all data, ~120 GB
python scripts/download.py data --out ./ferro_data --dataset d1 --split test_interior test_edge   # D1 test set only
python scripts/download.py models --out "$SCRATCH"                       # all checkpoints + Fourier SSMs, ~5 GB
```

`python scripts/download.py data --help` lists the other options (`--dataset`, `--split`, `--limit`, `--metadata-only`).

## Evaluate the trained models

Download the checkpoints into `$SCRATCH` and the dataset into `./ferro_data` as above, then:

```bash
export SCRATCH=/path/to/workdir                 # data, checkpoints and caches go here (~200 GB)
bash scripts/link_hf_data.sh ./ferro_data
python duhamel_no/build_train_cache.py          # training-set cache the data loader needs
python duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap --eval_only
```

## Train

```bash
python duhamel_no/fit_alpha_bank.py --task tdgl                                              # the Fourier SSM
python duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap   # Duhamel-NO
```

`scripts/train_all.sh` holds the command of every trained model of the paper, one per line. The full walkthrough
(each step, how long it takes, how hyperparameters were selected, regenerating the data with FerroX) is in
**[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

## Use the benchmark only

`benchmark/ferro_bench.py` loads the released data and computes the paper's metrics. It needs only numpy.

```python
import sys; sys.path.insert(0, "benchmark"); import ferro_bench as fb
fb.ROOT = "./ferro_data"
test = fb.metadata("d1", ["test_interior", "test_edge"])     # 256 held-out materials
true = fb.load("d1", test[0])                                 # (401, 10, 64, 64)
fb.score(my_rollout, true)                                    # corr and rel. L2 at frames 1 and 400
```

## Repository layout

```
duhamel_no/        the model (models.py), data loading (tasks.py), training/evaluation (run.py), metrics
scripts/           download.py, make_tables.sh, train_all.sh, evaluate_all.sh, link_hf_data.sh
results/           the result files behind every number in the paper
reproduce/         the scripts that turn results/ into the paper's numbers and tables, and the values it prints
benchmark/         standalone loader + metrics for the released dataset
data_generation/   FerroX inputs and job lists (D1, D2, hysteresis), material sampling, Allen–Cahn generator
docs/              REPRODUCE.md: the detailed step-by-step guide
```

## Citation

Citation information is withheld for double-blind review and will be added upon acceptance.

## License

Code and trained models: [BSD-3-Clause](LICENSE). Data: CC BY 4.0 (see the dataset card).
