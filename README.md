# microCT Rock Image Super-Resolution

This repository contains PyTorch code for super-resolution of micro-computed tomography images of rock samples. The workflow includes paired low-resolution/high-resolution image loading, preprocessing, paired data augmentation, hyperparameter optimization, cross-validation training, MSE/PSNR evaluation, runtime reporting, and generation of output images and plots.

## Repository structure

```text
configs/       experiment settings
data/          dataset folder skeleton
docs/          additional notes
microct_sr/    Python package with data loading, preprocessing, models, losses, metrics, training, and plotting
scripts/       command-line scripts for full experiments
quick_test.py  quick-test file
```

The `microct_sr/` folder contains the main Python code. The `scripts/` folder contains command-line files for running the experiments.

## Supported experiments

The repository follows the experiment setup used in the manuscript.

```text
COIFPM dataset:
  4x, 8x, and 16x super-resolution
  Methods: bicubic interpolation and simplified EDSR

DRSRD1 dataset:
  4x super-resolution
  Methods: bicubic interpolation, SRCNN, FSRCNN, EDSR, and RCAN
```

Porosity and pore-connectivity analyses were performed separately using Avizo and are not part of this Python workflow.

## Installation

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable installation allows the scripts to import the `microct_sr` package correctly.

## Data layout

The repository preserves the expected dataset folder structure. The full dataset image files are not committed to GitHub. To run the full experiments, place the local LR/HR image pairs in the folders shown below. LR and HR filenames must match.

```text
data/
  COIFPM/
    4x/
      hr/
      lr/
    8x/
      hr/
      lr/
    16x/
      hr/
      lr/

  DRSRD1/
    4x/
      hr/
      lr/
```

Example:

```text
data/COIFPM/4x/hr/sample_001.png
data/COIFPM/4x/lr/sample_001.png
```

The same filename should exist in both the `hr/` and `lr/` folders.

Dataset image files are excluded through `.gitignore`, while `.gitkeep` files preserve the folder structure.

## Quick test

A quick-test file is included to verify that the repository is installed correctly and that the main workflow can run.

Run:

```bash
python quick_test.py
```

The quick test first checks whether local COIFPM 4x image pairs are available in:

```text
data/COIFPM/4x/hr/
data/COIFPM/4x/lr/
```

If local COIFPM 4x image pairs are available, the quick test copies a small subset of matched LR/HR pairs into:

```text
outputs/quick_test_data/
```

and runs the test using that subset.

If local COIFPM 4x image pairs are not available, the quick test creates a small synthetic image-pair dataset automatically under:

```text
outputs/quick_test_data/
```

This allows the repository test to run without distributing the full research datasets.

The quick test checks installation, imports, image-pair loading, preprocessing, model construction, MSE loss calculation, PSNR calculation, cross-validation, hyperparameter optimization, runtime reporting, and output writing. It is intended as a repository functionality test and is not intended to reproduce the full manuscript results.

The quick test writes temporary files under:

```text
outputs/
```

The `outputs/` folder is ignored by Git.

## Running the COIFPM experiments

Run all COIFPM experiments:

```bash
python scripts/run_coifpm_experiments.py
```

This runs:

```text
COIFPM 4x:  bicubic interpolation + simplified EDSR
COIFPM 8x:  bicubic interpolation + simplified EDSR
COIFPM 16x: bicubic interpolation + simplified EDSR
```

To run one COIFPM scale manually:

```bash
python scripts/evaluate_bicubic.py --config configs/coifpm_4x.yaml
python scripts/train.py --config configs/coifpm_4x.yaml --models simplified_edsr
```

For 8x:

```bash
python scripts/evaluate_bicubic.py --config configs/coifpm_8x.yaml
python scripts/train.py --config configs/coifpm_8x.yaml --models simplified_edsr
```

For 16x:

```bash
python scripts/evaluate_bicubic.py --config configs/coifpm_16x.yaml
python scripts/train.py --config configs/coifpm_16x.yaml --models simplified_edsr
```

## Running the DRSRD1 experiments

Run all DRSRD1 experiments:

```bash
python scripts/run_drsrd1_experiments.py
```

This runs:

```text
DRSRD1 4x: bicubic interpolation + SRCNN + FSRCNN + EDSR + RCAN
```

To run manually:

```bash
python scripts/evaluate_bicubic.py --config configs/drsrd1_4x.yaml
python scripts/train.py --config configs/drsrd1_4x.yaml --models srcnn fsrcnn edsr rcan
```

## Hyperparameter optimization

The training scripts include Optuna-based hyperparameter optimization. The search includes learning rate and batch size. The HPO settings are defined in the YAML files under `configs/`.

Example:

```yaml
hpo:
  enabled: true
  n_trials: 2
  epochs: 5
  batch_sizes: [8, 16, 32]
  lr_min: 0.00001
  lr_max: 0.001
```

For a longer hyperparameter search, increase `n_trials` and `epochs` in the corresponding configuration file.

## Outputs

The scripts write generated outputs under the `outputs/` folder. This folder is ignored by Git by default.

Typical output files include:

```text
summary.csv
fold_summary.csv
per_image_metrics.csv
hpo_trials.csv
best_hyperparameters.json
training.log
optimization_history.png
best_trial_training_loss.png
best_trial_validation_loss.png
best_trial_validation_psnr.png
training_loss_all_folds.png
validation_loss_all_folds.png
validation_psnr_all_folds.png
avg_training_loss.png
avg_validation_loss.png
avg_validation_psnr_all_folds.png
training_vs_validation_mse.png
psnr_distribution_boxplot.png
```

For each fold, the code saves matched LR, HR, bicubic, and super-resolved images using the original image name. Example:

```text
output_images_fold_0/
  lr_image_sample_001.png
  hr_image_sample_001.png
  bicubic_image_sample_001.png
  super_res_image_sample_001.png
```

The same structure is repeated for the remaining folds:

```text
output_images_fold_1/
output_images_fold_2/
output_images_fold_3/
output_images_fold_4/
```

Generated outputs are not committed to GitHub. They can be recreated by running the corresponding scripts.

## Evaluation metrics

The main metrics are:

```text
MSE
PSNR
runtime
```

MSE is used as the training loss. PSNR is computed from the MSE between the generated super-resolved image and the corresponding HR image.

## Data augmentation

Paired geometric augmentation is applied so that the LR and HR images receive the same transformation. The augmentation options are configured in the YAML files:

```yaml
augmentation:
  enabled: true
  apply_to_train: true
  apply_to_validation: true
  horizontal_flip: true
  vertical_flip: true
  rot90: true
```

## Notes on data splitting

The code pairs LR and HR images by filename before splitting. Hyperparameter optimization and final cross-validation use separate index sets when HPO is enabled. K-fold splitting is performed at the paired-image level.

Validation is performed using `model.eval()` and `torch.no_grad()`, and optimizer updates occur only during training batches.

For datasets with highly correlated adjacent CT slices, group-based splitting by rock sample or volume can be added if required.

## Associated manuscript

This repository accompanies the manuscript:

```text
Application of Artificial Neural Networks for Enhancing Resolution of Micro-Computed Tomography Images from Rock Samples
```

Formal citation information can be added after publication.

## Tracked and ignored files

The repository is intended to track source code, configuration files, documentation, and the dataset folder skeleton.

The following files and folders are intended to be tracked:

```text
README.md
quick_test.py
requirements.txt
pyproject.toml
LICENSE
.gitignore
configs/
data/
docs/
microct_sr/
scripts/
```

The following files and folders are intended to be excluded by `.gitignore`:

```text
outputs/
results/
checkpoints/
__pycache__/
*.pyc
*.pt
*.pth
*.ckpt
data/**/*.png
data/**/*.jpg
data/**/*.jpeg
data/**/*.tif
data/**/*.tiff
data/**/*.bmp
```

These exclusions keep generated outputs, result files, model checkpoints, cache files, and dataset image files out of the GitHub repository. The dataset folder structure is preserved using `.gitkeep` files.

## License

This code is released under the MIT License. See the `LICENSE` file for details.