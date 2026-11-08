# Duhamel Neural Operator

Code for **"Duhamel Neural Operator for Learning Phase-Field Dynamics"**

Duhamel-NO advances a phase field one frame at a time. A **Fourier SSM**, measured from simulations, moves every Fourier
mode by its exact linear factor, and a **U-shaped neural operator** learns the nonlinear correction.

## Quick start: reproduce the paper's numbers (5 minutes, no GPU, no download)

```bash
git clone https://github.com/matthew-ju/Duhamel-Neural-Operator.git
cd Duhamel-Neural-Operator
pip install -r requirements.txt
bash scripts/make_tables.sh
```

This rebuilds every number in the main text, Table 1, Table 2 and Figure 3 from the included result files and compares
them with the published paper. The last line should be `ALL MATCH`.

## Train and evaluate

```bash
export SCRATCH=/path/to/workdir                 # data, checkpoints and caches go here (~200 GB)

# 1. download the data (Hugging Face) and link it into $SCRATCH
pip install -U huggingface_hub
hf download matthewju/ferroelectric-tdgl-bench --repo-type dataset --local-dir ./ferro_data
bash scripts/link_hf_data.sh ./ferro_data

# 2. train one model (Duhamel-NO on the ferroelectric data)
python duhamel_no/build_train_cache.py
python duhamel_no/fit_alpha_bank.py --task tdgl
python duhamel_no/run.py --mode final --task tdgl --model duhamel_uno --seed 0 --lr 0.0003 --wrap
```

`scripts/train_all.sh` and `scripts/evaluate_all.sh` hold every training and evaluation run of the paper, one per line.
The full walkthrough (each step, how long it takes, how hyperparameters were selected, regenerating the data with
FerroX) is in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

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
data_generation/   FerroX inputs and job lists (D1, D2), material sampling, Allen–Cahn generator
docs/              REPRODUCE.md: the detailed step-by-step guide
```

## Citation

```bibtex
@misc{ju2026duhamel,
  title     = {Duhamel Neural Operator for Learning Phase-Field Dynamics},
  author    = {Ju, Matthew and Tang, Yingheng and Wang, Kang'an and Sha, Haozhi},
  year      = {2026},
  url       = {https://github.com/matthew-ju/Duhamel-Neural-Operator}
}

@misc{ju2026ferrobench,
  title     = {Ferroelectric {TDGL} Phase-Field Benchmark},
  author    = {Ju, Matthew and Tang, Yingheng and Wang, Kang'an and Sha, Haozhi},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/11169},
  url       = {https://huggingface.co/datasets/matthewju/ferroelectric-tdgl-bench}
}
```

## License

Data: CC BY 4.0 (see the dataset card). Code: to be added after LBNL Intellectual Property Office approval.
