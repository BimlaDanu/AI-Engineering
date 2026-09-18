# MNIST digit classification with a multi-layer perceptron

![Python](https://img.shields.io/badge/python-3.11-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)
![Lightning](https://img.shields.io/badge/Lightning-2.2%2B-792ee5)
![Lint](https://img.shields.io/badge/lint-ruff-261230)
![Types](https://img.shields.io/badge/types-mypy%20strict-1f5082)
[![check](https://github.com/TuringCollegeSubmissions/bidanu-NLP.AI.1.5/actions/workflows/check.yml/badge.svg?branch=main&event=push)](https://github.com/TuringCollegeSubmissions/bidanu-NLP.AI.1.5/actions/workflows/check.yml)

A fully connected classifier for the Kaggle
[digit-recognizer](https://www.kaggle.com/competitions/digit-recognizer) dataset.

The training loop is plain PyTorch: forward pass, loss, backward pass and
optimizer step are written out. A Lightning wrapper reuses the same model, loss
and optimizer. Hyperparameters come from a seeded screening study over ten
factors, and the test split is read once, at the end.

The report notebook is `notebooks/mnist_mlp_report.ipynb`, executed start to
finish. Every number in it comes from a function in `src/` that the test suite
covers. `notebooks/mnist_mlp_gpu_report.ipynb` is a GPU session's copy of the
same notebook and is only partly executed, so read the first one. The derivations
behind the design are in
[`Mathematical_theoretical_analysis.md`](Mathematical_theoretical_analysis.md).

---

## Contents

[Results](#results) · [Quick start](#quick-start) · [Data](#data) ·
[Commands](#commands) · [Running the notebook](#running-the-notebook) ·
[Running on a cloud GPU](#running-on-a-cloud-gpu) ·
[Developer setup](#developer-setup) · [Technology stack](#technology-stack)

Below those, in order: the project tree, the method and the pipeline, the
screening study behind the hyperparameters, the improvements measured on top of
the baseline, what the project covers, and what is left to do.

---

## Results

The run named `final`, scored once on the held-out test split:

| Metric | Value |
|---|---|
| **Test accuracy** | **0.9856** |
| Macro F1 | 0.9855 |
| Expected calibration error | 0.0787 |
| Parameters | 1,199,114 |
| Test rows | 8,400, untouched until this run |

Its configuration, every field chosen by the screening study rather than by
hand: `512-512-512-512` hidden units, ReLU, batch normalisation, dropout 0.5,
Adam at `3e-3`, weight decay `1e-3`, batch size 32, a cosine schedule,
augmentation on, and the label-smoothing loss. Patience is off for the reported
runs, so it trained all 150 epochs and reached its best validation accuracy,
0.9843, at the last one. Reproduce it with:

```bash
make train ARGS="--name final --hidden-sizes 512 512 512 512 --dropout 0.5 \
  --optimizer adam --lr 3e-3 --weight-decay 1e-3 --batch-size 32 --augment \
  --loss label_smoothing --scheduler cosine --max-epochs 150 --patience 150"
make evaluate ARGS="--name final"
```

Two entries in that table need a word of explanation.

**The calibration error is high, and label smoothing is why.** Every bin of the
reliability table is under-confident: in the top bin the mean confidence is
0.9209 against an accuracy of 0.9987. Label smoothing buys accuracy by refusing
to let the logits grow confident, and the ECE is the bill for it. Temperature
scaling fitted on validation would recover the calibration without moving the
accuracy.

**Validation accuracy was still rising at epoch 150**, so the epoch limit ended
this run rather than the model converging. Training accuracy finishes at 0.9406,
*below* validation, which is what dropout 0.5 and a fresh warp on every training
batch look like from the inside: the training batches are harder than the clean
validation set.

**The study behind those choices** is `reports/experiment_summary.md`: 10
factors, 44 levels, 3 seeds, 132 runs. Every level is a mean over the seeds with
its standard deviation beside it, and a gap smaller than that spread is not a
result. Section 4 of the notebook plots it; section 5 feeds the winning level of
each factor into the confirmation run above and scores it on validation before
the test split is touched.

Two factors moved the result by more than the seed spread. Augmentation is the
large one, 0.9757 to 0.9896 against a standard deviation of 0.0001, which is
what an architecture with no built-in translation invariance should show. Label
smoothing is the other, 0.9757 to 0.9812. No level of activation, scheduler or
weight decay beat the baseline by more than one standard deviation, so on this
dataset none of the three bought anything. They can still cost: tanh and sigmoid
land at 0.9700 and 0.9704, five standard deviations below ReLU.

`reports/test_report.md` has a fixed name and describes whichever checkpoint was
evaluated last, so it may not be the run above. The run name is printed at the
top of it.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/). Everything else installs into
a project-local `.venv`.

```bash
make sync                                   # build .venv from uv.lock
make check                                  # lint + typecheck + the test suite
make train ARGS="--scheduler cosine --augment"
```

`make help` lists every target; `python -m src.train --help` lists every flag.
Nothing in this project writes outside the project directory.

## Data

```bash
make data        # download the competition CSVs into data/raw/, if not already there
make data-check  # row counts, what cleaning removes, split sizes
```

**A clone already has the data.** `data/raw/train.csv.gz` and
`data/raw/test.csv.gz` are committed at 8.5 MB and 5.6 MB against 73 MB and
49 MB, and pandas reads them without unpacking. `make data` is a no-op when
they are present and needs no Kaggle credentials.

`make data-force` downloads regardless; `make data-gz` rebuilds the compressed
copies afterwards. Downloading needs the competition rules accepted on the
website and the CLI authenticated with `kaggle auth login` (2.2.x uses OAuth or
a single API token, not the old `KAGGLE_USERNAME`/`KAGGLE_KEY` pair). Failing
that, download `train.csv` from the
[data page](https://www.kaggle.com/competitions/digit-recognizer/data) and drop
it in `data/raw/`.

Everything reported here comes from `train.csv`. The competition's `test.csv`
carries no labels, so the held-out test split is carved out of `train.csv`
instead. `test.csv.gz` is committed for a Kaggle submission; no module here
reads it.

## Commands

```bash
# one model; every factor the study varies has a flag
make train ARGS="--hidden-sizes 512 256 --optimizer adamw --lr 1e-3 --scheduler cosine"

# --seed moves the weights and the batch order; --split-seed moves the partition,
# so repeating a run over --seed compares models rather than partitions
make train ARGS="--seed 1"

# the study. 132 runs on a CUDA GPU, 20 without one; the machine decides
make experiments

# force either size, wherever you are
make experiments ARGS="--quick"   # 20 runs, seconds
make experiments ARGS="--full"    # 132 runs, about 95 minutes on a laptop

# score a saved checkpoint on the held-out test split
make evaluate ARGS="--name final"

# seed ensemble vs test-time augmentation
make ensemble ARGS="--seeds 0 1 2 3 4"

# the same data and loop, but a CNN, for the comparison the report needs
make compare-cnn
```

The device is picked automatically in the order cuda, mps, cpu. `--device cpu`
forces the CPU, which pairs with `--quick` on a laptop.

| Output | Written by |
|---|---|
| `checkpoints/<name>.pt` | `train`: weights, config, best epoch |
| `reports/history_<name>.csv` | `train`: per-epoch loss, accuracy, learning rate, gradient norms |
| `reports/experiments.csv` | `experiments`: one row per run |
| `reports/experiment_summary.md` | `experiments`: mean and spread per level |
| `reports/experiments-quick.csv`, `experiment_summary-quick.md` | `experiments --quick`: the short sweep, kept apart |
| `reports/test_report.md` | `evaluate`: test metrics, confusion, calibration |
| `reports/ensemble.md` | `ensemble`: members vs ensemble vs TTA |
| `reports/mlp_vs_cnn.md` | `compare-cnn`: accuracy and parameters, side by side |
| `reports/figures/*.png` | `evaluate`: curves, confusion matrix, error grid |

A sweep writes its rows as they finish, so an interrupted run still leaves
usable results, and a run that raises is recorded with its error rather than
ending the sweep.

## Running the notebook

```bash
make notebook              # jupyter lab, opened on notebooks/
make notebook-run          # run every cell start to finish and save the outputs
make notebook-run QUICK=1  # the same, at the reduced epoch counts
```

The cell holding the imports defines `QUICK`, set from the hardware rather than
by hand:

```python
_forced = os.environ.get('MNIST_MLP_QUICK')
QUICK = _forced not in {'0', 'false'} if _forced is not None else not full_study_is_affordable()
```

*Run all* then finishes in minutes on a laptop and produces the reported numbers
on a CUDA GPU, with no edit in between. It reads the same function
`make experiments` does, so the notebook and the command line never disagree.

`MNIST_MLP_QUICK` overrides the hardware when you mean to: `0` asks for the full
study on a machine the check rules out, `1` asks for the short one. That is what
`make notebook-run` sets, and it is the only way to execute the whole notebook
without leaving a browser tab open for hours. The outputs a reader sees in the
committed file come from that command, not from cells run by hand in whatever
order.

| | `QUICK = True` | `QUICK = False` |
|---|---|---|
| baseline and each sweep run | 8 epochs | 100 |
| confirmation run | 12 | 150 |
| each ensemble member, and the CNN | 6 | 100 |
| the sweep | 2 levels per factor, 1 seed, **20 runs** | 10 factors, 44 levels, 3 seeds, **132 runs** |
| writes to | `experiments-quick.csv`, `final-quick.pt` | `experiments.csv`, `final.pt` |

Those epoch counts are ceilings, not targets; what ends a run is `patience`.
The notebook sets it two ways. The sweep keeps patience at 10, since its 132
runs exist to be compared with each other and the tails are paid 132 times. The
six runs whose numbers are quoted switch it off, so their schedules anneal over
the full epoch limit they were given and the curves run past the overfitting
turn.

The short sweep gives up the seeds, which is the expensive part, so there is no
spread left to read a gap against.

A quick run writes to its own file names, on the command line as much as here,
so trying the pipeline out cannot cost you a full run's checkpoint or sweep
results. The exception is `reports/test_report.md`, which has a fixed name;
regenerate it with `make evaluate ARGS="--name final"`.

On Apple Silicon `select_device` picks MPS, which has no float64 kernels, so
`accumulator_dtype` sums each epoch's running totals in float32 there and in
float64 everywhere else. That is the only place the backend changes anything.

## Running on a cloud GPU

The sweep is 132 short runs: about 45 minutes on a T4 and an hour and a half on
an Apple-silicon laptop. Measured, not estimated: 95 minutes of training over the
132 runs here, against a T4 2.1x faster on the identical run.

The notebook's first two cells handle the session: they unpack an uploaded
bundle into a directory of its own, find the project rather than assuming where
it landed, move it somewhere writable on Kaggle, and install only the packages
the image is missing.

**Getting the code in.** Set the runtime to a GPU first. On Colab that is
*Runtime, Change runtime type, T4 GPU*; on Kaggle, *Settings, Accelerator,
GPU*. Then:

```bash
make bundle      # one 15 MB zip: exactly the files git tracks
```

Upload that, not the project folder, which is over a gigabyte because `.venv` is
in it. The zip holds what `git ls-files` lists: `src/`, `notebooks/`, the
Makefile, the lock file, both `data/raw/*.csv.gz`. Nothing gitignored reaches
it, so no `.env`, no `reports/`, no checkpoints, no caches. It zips the working
copy, not the last commit, so uncommitted edits travel with it.

Upload the zip, not the notebook alone: every cell imports from `src/`. On
Colab, drag it into the Files pane (folder icon, left) and run the notebook;
the first cell extracts it. On Kaggle, add it through *+Add Input, Datasets, New
Dataset* as a private dataset; Kaggle mounts datasets read-only and the run
writes `reports/`, `checkpoints/` and `data/`, so the bootstrap copies the
project to `/kaggle/working`. Cloning works where the session can reach the
repository, which is private and needs a token.

Re-uploading later needs the old copy gone first. The unpack cell stops when it
finds an already-extracted tree, so a new zip beside an old directory is
ignored. Delete the directory, or start a fresh runtime.

**Getting `train.csv`.** Nothing to do: the bundle and the repository both carry
`data/raw/train.csv.gz`, and `load_raw_train` falls back to it. Confirm it
before spending GPU time with `python -m src.data`, which prints the cleaning
report and the split sizes.

On Kaggle you can instead add the competition through *Add Input, Competitions,
Digit Recognizer*, which mounts it read-only. The bootstrap detects that mount,
and every entry point takes it with `--csv`:

```bash
python -m src.experiments --csv /kaggle/input/digit-recognizer/train.csv
```

**Getting the results back out.** The session disk is temporary, and four small
files are what you need to keep: `reports/experiments.csv`,
`reports/experiment_summary.md`, `reports/test_report.md` and
`checkpoints/final.pt`. On Colab, mount Drive and copy `reports/` and
`checkpoints/` into it; on Kaggle, anything under `/kaggle/working` is offered
as an output once the notebook is committed. Dropping those two folders into a
local checkout lets the notebook's later sections run without retraining, and
`python -m src.evaluate --name final` needs only the checkpoint.

**Splitting the sweep.** A twelve-hour limit is generous; a dropped connection
is not. Splitting by factor turns one long job into four restartable ones, each
writing its own file:

```bash
python -m src.experiments --factors optimizer learning_rate batch_size --out reports/experiments_part1.csv
python -m src.experiments --factors activation dropout hidden_sizes --out reports/experiments_part2.csv
python -m src.experiments --factors loss scheduler --out reports/experiments_part3.csv
python -m src.experiments --factors weight_decay augment --out reports/experiments_part4.csv
```

Those four cover all ten factors, and a test fails if a factor is added to
`FACTORS` without being added to a chunk. Concatenate the parts into
`reports/experiments.csv` when they are all in; `summarise` and `write_summary`
aggregate the combined file exactly as they would a single run. When compute time
is the problem rather than the connection, `--seeds 0 1` cuts the sweep by a
third at the cost of a wider error bar, and `--max-epochs 20` shortens every
fit. Both change what the numbers mean, so say which was used.

## Project structure

```
.
├── Makefile                        every command in the project; start with `make help`
├── pyproject.toml                  dependencies plus the ruff, mypy and pytest settings
├── uv.lock                         exact pinned versions; committed on purpose
├── .python-version                 interpreter uv installs (3.11.3)
├── .gitignore                      keeps the raw CSVs, checkpoints and figures out of git
├── README.md                       this file
├── Mathematical_theoretical_analysis.md   the derivations behind the design
├── .github/
│   └── workflows/
│       └── check.yml               CI: the same `make check` gate, on main and on PRs
│
├── src/
│   ├── __init__.py                 package marker
│   ├── config.py                   frozen dataclasses holding every setting of a run
│   ├── utils.py                    project paths, seeding, device selection
│   ├── data.py                     load, clean, split 60/20/20, build DataLoaders
│   ├── models.py                   the configurable MLP and its weight initialisation
│   ├── augment.py                  random shifts and rotations, applied on-device
│   ├── training.py                 the plain PyTorch loop: losses, schedules, early stopping
│   ├── train.py                    CLI for a single run; checkpoint save and load
│   ├── experiments.py              the screening sweep and its summary tables
│   ├── evaluate.py                 test metrics, confusion, calibration, figures
│   ├── ensemble.py                 seed ensembling and test-time augmentation
│   ├── reference_cnn.py            a small CNN, for the MLP-versus-CNN comparison
│   └── lightning_module.py         Lightning wrappers around the same model and data
│
├── tests/                          `make test-fast` skips the ones that train
│   ├── __init__.py                 makes tests importable as a package (mypy needs it)
│   ├── conftest.py                 synthetic stand-in for train.csv, shared fixtures
│   ├── test_data.py                cleaning rules, split proportions, no leakage
│   ├── test_models.py              shapes, activations, logits, initialisation
│   ├── test_augment.py             warp geometry, background fill, per-image randomness
│   ├── test_training.py            optimizers, losses, schedules, early stopping, seeds
│   ├── test_train_cli.py           flag wiring, checkpoint round trip, evaluation path
│   ├── test_experiments.py         factor coverage, run naming, aggregation
│   ├── test_evaluate.py            confusion, confident mistakes, calibration, figures
│   ├── test_ensemble.py            probability averaging, TTA, member independence
│   ├── test_reference_cnn.py       CNN shapes, parameter count, translation behaviour
│   ├── test_lightning_module.py    Lightning wiring and parity with the plain loop
│   ├── test_utils.py               paths stay in the project, seeding, device selection
│   ├── test_notebook.py            the report notebook: parses, imports, stays in step
│   └── test_docs.py                docs match the commands, flags and files they quote
│
├── notebooks/
│   ├── mnist_mlp_report.ipynb      the report: data, sweep, final model, conclusions
│   └── mnist_mlp_gpu_report.ipynb  the same report from a GPU session; see the note below
│
├── data/
│   ├── README.md                   what is committed, and how to obtain the rest
│   ├── raw/                        train.csv.gz and test.csv.gz, committed
│   └── processed/                  intermediate files, if any; not committed
│
├── reports/                        generated tables and markdown summaries
│   └── figures/                    generated plots; only the final run's are committed
│
└── checkpoints/                    saved model weights; not committed
```

## Developer setup

**Environment.** One uv-managed `.venv`. Runtime dependencies sit in
`pyproject.toml` under `[project]`, the dev tools under `dev`, and the Kaggle
client under `data` so it is not needed to train.

```bash
make sync      # create or update .venv from the lock file
make sync-ci   # install exactly what uv.lock pins, no re-resolution
make lock      # re-resolve after editing pyproject.toml
```

The Makefile calls uv as `env -u VIRTUAL_ENV uv`, so an environment already
activated in the shell cannot shadow the project's own.

**Quality gate.** `make check` is the definition of done, and CI runs the same
three tools on every push to `main` and on every pull request:

| Tool | Configured in | Checks |
|---|---|---|
| ruff | `[tool.ruff]` | PEP8, imports, naming, bugbear, docstrings; line length 100 |
| mypy | `[tool.mypy]` | `src/` and `tests/`, with `disallow_untyped_defs` |
| pytest | `[tool.pytest.ini_options]` | `testpaths = tests`; `slow` marks the ones that train |

```bash
make format     # auto-format and auto-fix
make lint       # check without writing
make typecheck
make test
make test-fast  # skip the training tests, for a tight edit loop
make check      # all three
```

**Working locally.** `src/utils.py:full_study_is_affordable` is the single place
that decides the sweep size, and the notebook's `QUICK` switch reads the same
function, so [Commands](#commands) and
[Running the notebook](#running-the-notebook) describe one mechanism rather than
two. MPS counts as a laptop there. The short sweep subsamples the data and
shrinks the network as well as the epoch count: it proves the pipeline runs, and
with one seed it has no error bars, so its numbers are not reportable.

```bash
make train ARGS="--max-rows 3000 --max-epochs 3 --device cpu"   # one small model, fast
```

**Conventions.**

- Every setting lives in a dataclass in `src/config.py`. A new knob is a field
  there, not a parameter threaded through five functions.
- Modules expose functions; the CLIs are thin wrappers over them, so the
  notebook calls exactly the code the tests cover.
- `src/utils.py` owns all paths. Nothing builds one by hand.
- Public functions carry type hints and a docstring saying what they return and
  what they raise.
- The documentation is tested. `tests/test_docs.py` checks every command, flag
  and path quoted here against the code, so a renamed flag fails the suite
  rather than failing when the line is copied out.

**Adding a factor to the sweep.** Add the field to `ModelConfig` or
`TrainConfig`, add its levels to `FACTORS` in `src/experiments.py`, and, if it
belongs to the model, its name to `MODEL_FACTORS`. `configuration_for` handles
the rest, and `test_every_training_field_the_sweep_varies_has_a_flag` fails
until the CLI can set it too.

## Method

**Model.** `Flatten → [Linear → BatchNorm → Activation → Dropout] × n → Linear`.
The last layer returns logits; the loss applies the softmax. An empty hidden
stack gives a linear classifier, the floor the sweep is read against.

**Losses.** `cross_entropy`, `label_smoothing`, `nll`, and squared error against
a one-hot target. The last two are there for the comparison: `nll` is the same
objective written out, and `mse` converges slowly on this problem.

**Optimizers.** `sgd`, `sgd_momentum`, `adam`, `adamw`, `rmsprop`. The optimizer
and the learning rate are separate factors, since a rate does not transport
between them.

**Initialisation.** Kaiming for the ReLU family, Xavier for the saturating
activations, chosen from the configured activation rather than set by hand.

**Regularisation.** Dropout, weight decay and batch normalisation, all three
swept.

**Schedules.** `none`, `step`, `cosine`, `onecycle`. Each spreads itself over
`max_epochs`, so a run that stops early ends part way through its schedule. The
rate every epoch trained at is written to the history, so this can be read off
the run rather than assumed.

**Diagnostics.** `reports/history_<name>.csv` records the gradient norm of every
weight matrix each epoch beside the loss and the accuracy, so a vanishing
gradient is something a run shows rather than something the README asserts.

**Calibration.** `make evaluate` reports expected calibration error over
confidence bins alongside accuracy and macro F1.

Why each of these is the choice it is, derived rather than asserted:
[`Mathematical_theoretical_analysis.md`](Mathematical_theoretical_analysis.md).

## Pipeline

**Cleaning, before splitting.** `src/data.py` drops rows with missing values, a
pixel outside $[0,255]$ or a label outside $[0,9]$, repeated images, and images
of a single constant value. Repeated images carrying *different* labels lose
every copy rather than one. Cleaning runs before the split, and a test asserts
that no image appears in two splits. On the competition file as shipped, all
five rules find nothing; the cleaning report prints the counts, so that is
visible rather than assumed.

**Split.** Stratified 60 / 20 / 20 on a fixed seed, so two runs compare models
rather than partitions. The mean and standard deviation used to standardise the
pixels come from the training split alone.

**Training.** Early stopping watches the validation loss; the weights kept are
those of the best validation *accuracy*, the headline metric. Per-epoch totals
accumulate on the device and are read back once, because a `.item()` inside the
batch loop stalls the GPU on every step.

**Evaluation.** Nothing is selected on the test split. Hyperparameters, epoch
and checkpoint are all chosen on validation, and the test rows are read only to
score what validation already picked. `src/evaluate.py` does the scoring, and
`src/ensemble.py` and `src/reference_cnn.py` keep to the same rule. The report
covers accuracy, per-class precision, recall and F1, the confusion matrix with
its largest off-diagonal entries, a reliability table, and a grid of the
mistakes the model was most confident about.

## Hyperparameter study

A full grid over ten factors is not affordable, so `src/experiments.py` runs a
screening design: one factor varied at a time from a fixed baseline, every
configuration repeated over three seeds.

| Factor | Levels |
|---|---|
| optimizer | sgd, sgd_momentum, adam, adamw, rmsprop |
| learning_rate | 1e-4 … 1e-2 |
| batch_size | 32, 64, 128, 256, 512 |
| activation | relu, leaky_relu, gelu, tanh, sigmoid |
| dropout | 0.0, 0.1, 0.2, 0.3, 0.5 |
| hidden_sizes | (128,) … (512, 512, 512, 512) |
| loss | cross_entropy, label_smoothing, nll, mse |
| scheduler | none, step, cosine, onecycle |
| weight_decay | 0.0, 1e-5, 1e-4, 1e-3 |
| augment | off, on |

Results are read as mean ± standard deviation across seeds. A gap smaller than
the seed-to-seed spread is not a result, so every configuration is repeated
instead of reporting the best single run.

Two couplings the design cannot see: the learning rate with the optimizer, and
the learning rate with the batch size. A confirmation run combines the winning
levels and checks them together. `best_levels` reads those winners off the
summary table and `configuration_from` applies them, so that run is derived from
the study rather than typed out by hand.

The `scheduler` row carries one caveat. Sweep runs keep `patience=10`, and every
schedule spreads itself over `max_epochs`, so a level that stops early is scored
part way through its own annealing. The comparison is between schedules as they
behave under early stopping, not between fully annealed ones; the reported runs
train theirs out.

## Improvements beyond the baseline

Three improvements, implemented and measured rather than left as suggestions.

**Augmentation** (`src/augment.py`). Each training batch is warped on the device
by a per-image affine map, with the rotation angle and the two offsets drawn
fresh every epoch. It is a sweep factor, so its effect is measured against the
same seed spread as everything else.

**Ensembling and TTA** (`src/ensemble.py`). Class probabilities averaged over
models that differ only by seed, and the same averaging over several warped
copies of one input. `reports/ensemble.md` puts the members, the ensemble and
TTA in one table.

**The CNN comparison** (`src/reference_cnn.py`). Same data, same loop, same epoch
limit, so the architectural gap is a measured number rather than an assertion.

## What the project covers

| Capability | Where |
|---|---|
| Data from Kaggle | committed as `data/raw/train.csv.gz`; `make data` re-fetches |
| 3-way split, 60 / 20 / 20 | `src/data.py`, stratified and seeded |
| Customizable training pipeline | `src/config.py` + `src/train.py`, one flag per field |
| Vary optimizers, batch sizes, activations, learning rates, dropout, architectures, losses | `src/experiments.py`, all seven plus three more |
| Best model evaluated on test | `src/evaluate.py`, run once |
| PyTorch and PyTorch Lightning | `src/training.py` and `src/lightning_module.py` |
| GPU training | device auto-selection; the notebook's cloud bootstrap |
| Cleaning: duplicates, corruptions | `src/data.py`, reported by `make data-check` |
| Improvements beyond the baseline | implemented and measured, see above |
| Code quality, PEP8 | `make check`, and CI on every push to `main` |

## Further work

Ordered by expected gain over the MLP baseline:

1. **Elastic distortion** on top of the shifts and rotations. It helped most in
   the original LeCun results, and it is the next step now that the warp
   machinery exists.
2. **A joint search** over the factors the screening design cannot see together:
   the learning-rate/batch-size and learning-rate/optimizer interactions.
3. **A scheduler sweep without early stopping**, so the schedules are compared
   fully annealed rather than cut short.
4. **Temperature scaling** fitted on validation. The ECE is measured but
   nothing is done about it.
5. **Confidence intervals** on the test accuracy, from a bootstrap over the
   test split, so two models can be compared with something better than
   eyeballing.
6. **Switching to a CNN**, which the comparison above quantifies. It is the
   ceiling this project works under.

## Technology stack

PyTorch does the modelling and Lightning wraps it. Everything else is either
data handling, reporting, or the quality gate. The versions below are the floors
`pyproject.toml` pins; `uv.lock` fixes the exact build and `make versions`
prints what is installed.

| Tool | Version | What it does here |
|---|---|---|
| Python | 3.11.3 | pinned in `.python-version`, installed by uv |
| PyTorch | ≥ 2.2 | layers, autograd, optimizers, and the loop in `src/training.py` |
| Lightning | ≥ 2.2 | the same model and optimizer under its own loop, `src/lightning_module.py` |
| pandas | ≥ 2.2 | reads the CSVs, writes every report table |
| NumPy | ≥ 1.26 | the pixel arrays between pandas and torch |
| scikit-learn | ≥ 1.4 | the stratified split, the per-class report, the confusion matrix |
| matplotlib | ≥ 3.8 | training curves, confusion matrix, error grid |
| tabulate | ≥ 0.9 | what `DataFrame.to_markdown` needs to write those tables |
| uv | — | resolves and builds `.venv`; `uv.lock` pins it |
| Ruff | ≥ 0.5 | lint and format |
| mypy | ≥ 1.10 | types, with `disallow_untyped_defs` |
| pytest | ≥ 8.0 | the suite, this README included |
| JupyterLab | ≥ 4.2 | the report notebook |
| kaggle | ≥ 1.6 | optional, and only to re-download the competition CSVs |
| GitHub Actions | — | runs `make check` on pushes to `main` and on pull requests |

Training runs on CUDA, Apple MPS or the CPU, picked in that order at runtime by
`select_device`. Two packages are declared in `pyproject.toml` and imported
nowhere: `torchmetrics` and `seaborn`.
