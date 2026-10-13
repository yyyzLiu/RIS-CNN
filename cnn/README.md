# RIS CNN: total-field prediction

This component contains the CNN, 24,000 training samples, and a separate
test set for predicting the total tangential electric and magnetic fields of a
RIS. All required input and target values are embedded in the supplied
`.fld` samples. See the [repository citation section](../README.md#citation)
for the two papers describing the model.

## Install and run

Use Python 3.9 through 3.12. The scripts were checked with Python 3.9.12,
PyTorch 2.3.0, and NumPy 1.21.5. From this directory:

```bash
python -m pip install -r requirements.txt
python train.py
```

Paths are resolved relative to the scripts. PyTorch uses `cuda:0` when CUDA is
available and otherwise uses the CPU. The model is large; training is intended
for a CUDA GPU with a suitable PyTorch installation.

## Configure the experiment

Edit [config.json](config.json) to set the model and training parameters. Running
`python train.py` uses this file automatically. JSON does not support comments;
the fields are explained below. All fields shown in the supplied config are
required. Command-line overrides are applied after loading the config.

| Field | Default | Meaning |
| --- | --- | --- |
| `model.in_channels` | `5` | Input channel count: binary RIS state and real/imaginary incident Ex and Ey. Must remain 5 for the supplied data. |
| `model.out_channels` | `8` | Output channel count: real/imaginary total Ex, Ey, Hx, and Hy. Must remain 8 for the supplied data. |
| `model.features` | `[128, 256, 512, 1024]` | Feature widths of successive encoder blocks; the decoder uses them in reverse order. The bottleneck uses twice the last width. Each width must be an integer at least 16 for the attention blocks, and each width after the first must be twice the previous width. Larger widths increase memory use. |
| `model.dropout_rate` | `0.3` | Probability used by the model's `Dropout2d` layers during training; must be at least 0 and less than 1. |
| `data.samples_per_group` | `750` per group | Counts selected randomly without replacement for each array size and incident angle, before the training/validation split. See the example below. |
| `data.training_fraction` | `0.8` | Fraction of selected samples used for training, separately for each size. The remainder is used for validation. Must be between 0 and 1 and leave both sets nonempty. |
| `data.magnetic_scale` | `377.0` | Positive multiplier applied to magnetic target channels for training and test MSE. Divide predicted magnetic channels by this value to recover Hx and Hy. |
| `data.all_data` | `false` | When true, use all published training samples and ignore `samples_per_group` counts. The training/validation split still applies. |
| `training.seed` | `42` | Random seed for sample selection, training/validation splits, NumPy, and PyTorch. Must be an integer from 0 through 4294967295. GPU results can still vary across hardware and software versions. |
| `training.epochs` | `5000` | Maximum number of training epochs; a positive integer. Early stopping can end training sooner. |
| `training.batch_size` | `16` | Maximum number of samples in each training, validation, or test batch; a positive integer. Each batch contains a single array size. |
| `training.patience` | `100` | Stop after this many consecutive epochs without a lower validation loss; a positive integer. |
| `training.device` | `"auto"` | `auto` selects `cuda:0` when CUDA is available, otherwise CPU. Use `cpu` or a specific CUDA device such as `cuda:0` to select it explicitly. |
| `optimizer.name` | `"Adamax"` | PyTorch optimizer to use. Supported values are `Adamax`, `Adam`, `AdamW`, and `SGD`. |
| `optimizer.parameters` | `{"lr": 0.001}` | Keyword arguments passed to the selected optimizer. `lr` is the positive learning rate; additional arguments must be supported by that optimizer. |

MSE is the loss used by both training and evaluation. The supplied defaults are
listed above and described below.

### Choose samples for each size and angle

Edit `data.samples_per_group` in `config.json`. Its outer keys specify the array
size and inner keys specify the incident angle in degrees. Each value is the number of
samples to select from that group before splitting into training and validation.
Each training group contains 2,000 published samples, so counts cannot exceed 2,000.
For example, replace just `data.samples_per_group` with:

```json
{
  "10": {"0": 500, "10": 600, "20": 750},
  "16": {"0": 700, "30": 800}
}
```

Only the groups listed with positive counts are used. Samples are drawn randomly
without replacement once at the beginning of a run. The default configuration selects
750 samples at each of six angles for each of two sizes, totaling 9,000 samples.
Selection is reproducible with seed 42. Each size/angle group has an independent
random generator, so changing another group's count does not change its selection.

For each array size, the selected samples are split 80% for training and 20%
for validation by default (`data.training_fraction = 0.8`), using the configured
seed. Test samples are kept separate from both.
`results/run_<timestamp>/splits.json` records the exact selected paths and split.
Command-line options override the corresponding config settings for one run.
To change the seed or use a copy of the complete config file:

```bash
python train.py --seed 123
python train.py --config my_config.json
```

Use all 24,000 published training samples regardless of the configured counts:

```bash
python train.py --all-data
```

Validate selected training samples without creating a model:

```bash
python train.py --check-data
```

Add `--all-data` to check every training sample. Other controls include
`--epochs 5000`, `--batch-size 16`, `--patience 100`, and `--device cuda:0`.
Use `--no-all-data` to disable `data.all_data` when enabled in the config. Each run
records the effective configuration, including command-line overrides, in its
`config.json`.

## Total-field targets

The 5 input channels contain the binary RIS state and real/imaginary incident Ex
and Ey. The 8 output channels contain real/imaginary total Ex, Ey, Hx, and Hy.
Magnetic target channels are multiplied by `data.magnetic_scale` (default 377)
to retain the original total-field training convention; divide those prediction
channels by the same scale to obtain Hx and Hy.
Target construction reads the stored total-field columns directly.
See [data/README.md](data/README.md) for the column definitions and inventory.

The default architecture uses feature widths `[128, 256, 512, 1024]` and dropout 0.3.
Default training uses MSE, Adamax with learning rate 0.001, batch size 16, up to 5,000
epochs, and early stopping after 100 epochs without a validation improvement.
Epoch training and validation losses are means over batches. Seeded GPU runs
can still differ across hardware and software versions.

## Test set and outputs

The separate test set has 550 samples across 11 groups, including sizes 10, 12,
14, 16, 20, and 24. Check only the test files with:

```bash
python evaluate.py --check-data
```

Model evaluation is a separate step using `evaluate.py` and its evaluation config.
It reports total-field MSE using the configured magnetic scaling; the overall MSE
is weighted by the number of evaluated samples.

Each training run writes small records to `results/run_<timestamp>/`:
`losses.csv`, `splits.json`, `config.json`, and `summary.txt`.
The saved config reproduces the effective training settings; the summary records
start/end timestamps with time zones, elapsed time, completion reason, completed
and best epochs, losses, sample counts, device, and checkpoint location. Training
keeps the weights with the lowest validation loss in memory and always saves them once to
`model.pth` when training completes or early stopping triggers. The file records
the best epoch and validation loss. No checkpoints are written during training;
manually interrupting training before completion does not save a model. The
console prints the checkpoint path. Evaluate it separately using:

```bash
python evaluate.py --model-path results/run_<timestamp>/model.pth
```

### Evaluation configuration

Edit [evaluation_config.json](evaluation_config.json), then run
`python evaluate.py` without arguments or double-click `evaluate.py` on Windows.
Windows must associate `.py` files with a Python interpreter that has the required
packages installed. Paths in this JSON are relative to its own directory; absolute
paths are also accepted.

| Field | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | `null` | Path to the saved model or its run directory, for example `"results/run_<timestamp>/model.pth"`. When null, select the most recently saved checkpoint under this script's `results/run_*/` directory. |
| `data.dataset` | `"data/test"` | Directory containing size/angle folders, or one specific folder such as `"data/test/10by10_0deg"`. Prepared files must be named `sample_*.fld`; group folders must retain their size/angle naming convention. |
| `data.groups` | `null` | Evaluate every group when null, or select folder names such as `["10by10_0deg", "16by16_30deg"]`. The case limit applies to each selected group. |
| `data.samples_per_group` | `null` | Same size/angle/count mapping as training: outer keys are array sizes, inner keys are incident angles in degrees, and values are nonnegative case counts. Only listed positive counts are tested. Use null to test all available cases. |
| `batch_size` | `null` | Inherit `training.batch_size` from the config beside the checkpoint, or set a positive integer for evaluation only. |
| `device` | `null` | Inherit `training.device` from the config beside the checkpoint, or select `"auto"`, `"cpu"`, or a specific CUDA device. |

For example, set the evaluation `data` section to test 20 cases from the
10x10, 0-degree folder and 30 cases from the 16x16, 30-degree folder:

```json
"data": {
  "dataset": "data/test",
  "groups": null,
  "samples_per_group": {
    "10": {"0": 20},
    "16": {"30": 30}
  }
}
```

You can also point `data.dataset` directly to one size/angle folder and list that
size/angle in `samples_per_group`. Counts select samples randomly without
replacement using `training.seed` from the config beside the checkpoint.
Requesting more cases than a folder contains raises an error. If your dataset contains multiple folders with the same size and angle, the
count applies to each matching folder; use `data.groups` to choose a specific collection.

Evaluation automatically loads `config.json` from the selected checkpoint's
run directory. It supplies magnetic scaling, the sampling seed, and default
batch size/device; no separate training-config path is needed. Keep `config.json`
with `model.pth` when moving a trained model. Evaluation checks that the scale
matches the checkpoint's metadata. Model architecture is read from the checkpoint.
For `--check-data`, which does not require a model, these defaults come from the
main `config.json` beside the scripts.

Use `--evaluation-config my_evaluation.json` to select another evaluation config.
`--model-path`, `--batch-size`, and `--device` override the corresponding
config choices for a single run. The `EVALUATION_CONFIG` variable near the top of
`evaluate.py` selects the default evaluation config for launching the script.

Each evaluation saves its results under the selected model's run folder in a
`evaluation/` directory:

- `losses.csv`: per-group sample counts and total-field MSE.
- `summary.txt`: overall MSE, per-group losses, start/end times, elapsed time,
  sample counts, and checkpoint/dataset information.
- `evaluation_config.json`: effective settings, including command-line overrides,
  with paths resolved relative to this saved config so it can be reused.
- `samples.json`: exact tested filenames, relative to the selected dataset.

Each completed evaluation replaces the reports in this folder. `--check-data`
validates files without creating evaluation results.

The selected checkpoint path and test losses are printed, and the script exits
automatically when finished. A saved checkpoint is required; run `python train.py`
to create one.
Saving is always enabled and occurs once training completes or early stopping triggers.

The scripts train and evaluate total-field predictions only. No pretrained
weights are included. Generated results are ignored by Git.

## Citation

If you use this code or the training/test data in your research, please cite
either or both of the following papers, as relevant to your work:

1. Yuanzhi Liu and Costas D. Sarris, "A Convolutional Neural Network Based Method
   for Accurate Computation of Scattered Fields From Reconfigurable Intelligent
   Surfaces," *IEEE Transactions on Antennas and Propagation*, 2025.
   [DOI: 10.1109/TAP.2025.3532123](https://doi.org/10.1109/TAP.2025.3532123).
2. Yuanzhi Liu and Costas D. Sarris, "Optimization of Reconfigurable Intelligent
   Surface Codebooks Using a Mutual Coupling Aware CNN Model of Scattered Fields,"
   *IEEE Antennas and Wireless Propagation Letters*, 2025.
   [DOI: 10.1109/LAWP.2025.3585743](https://doi.org/10.1109/LAWP.2025.3585743).

BibTeX entries for both papers are provided in [CITATION.bib](../CITATION.bib).
