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
pip install -r requirements.txt
bash scripts/make_tables.sh
```

This rebuilds every number (77) and every table (10) of the paper, main text and appendix, plus the figures drawn from
results, from the included result files, and compares them with the submitted paper. The last line should be
`ALL MATCH`.

## Data and trained models

| what | where |
|---|---|
| D1, D2 and hysteresis data, with the paper's splits | [anonymous link](https://anonymous-hf.com/a/58mhxpc51v6u/) (dataset) |
| every trained model of the paper (109 checkpoints) and the measured Fourier SSMs | [anonymous link](https://anonymous-hf.com/a/58mhxpc51v6u/) (models) |

## Evaluate the trained models

Download the checkpoints into `$SCRATCH` and the dataset into `./ferro_data` from the links above, then:

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
scripts/           make_tables.sh, train_all.sh, evaluate_all.sh, link_hf_data.sh
results/           the result files behind every number in the paper
paper/             paper PDF and the scripts that turn results/ into its tables and figures
benchmark/         standalone loader + metrics for the released dataset
data_generation/   FerroX inputs and job lists (D1, D2, hysteresis), material sampling, Allen–Cahn generator
docs/              REPRODUCE.md: the detailed step-by-step guide
```

## Citation

Citation information is withheld for double-blind review and will be added upon acceptance.

## License

Code and trained models: [BSD-3-Clause](LICENSE). Data: CC BY 4.0 (see the dataset card).
