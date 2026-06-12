
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .preprocessing import tensor_to_image


def _finish(path: str | Path, show: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_optimization(study, output_path: str | Path, show: bool = False) -> None:
    trials = study.trials_dataframe()
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(trials['number'], trials['value'])
    plt.title('Optimization History')
    plt.xlabel('Trial')
    plt.ylabel('Validation PSNR')

    plt.subplot(1, 2, 2)
    lr_col = 'params_lr' if 'params_lr' in trials.columns else 'params_learning_rate'
    plt.scatter(trials[lr_col], trials['value'])
    plt.xscale('log')
    plt.title('PSNR vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation PSNR')
    _finish(output_path, show=show)


def plot_best_trial_curves(best_trial, output_dir: str | Path, show: bool = False) -> None:
    output_dir = Path(output_dir)
    train_losses = best_trial.user_attrs.get('train_losses', [])
    val_losses = best_trial.user_attrs.get('val_losses', [])
    val_psnrs = best_trial.user_attrs.get('val_psnrs', [])

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Best Trial Training Loss (Trial {best_trial.number})')
    plt.legend()
    _finish(output_dir / 'best_trial_training_loss.png', show=show)

    plt.figure()
    plt.plot(val_losses, label='Validation Loss (HPO)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss for Best Trial (HPO)')
    plt.legend()
    _finish(output_dir / 'best_trial_validation_loss.png', show=show)

    plt.figure()
    plt.plot(val_psnrs, label='Validation PSNR (HPO)', linestyle='-.')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Validation PSNR for Best Trial (HPO)')
    plt.legend()
    _finish(output_dir / 'best_trial_validation_psnr.png', show=show)


def plot_training_validation_curves(all_train_losses, all_val_losses, all_val_psnrs, output_dir: str | Path, show: bool = False) -> None:
    output_dir = Path(output_dir)
    line_styles = ['-', '--', '-.', ':', '-']

    plt.figure()
    for fold in range(len(all_train_losses)):
        plt.plot(all_train_losses[fold], line_styles[fold % len(line_styles)], label=f'Training Loss Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for All Folds')
    plt.legend()
    _finish(output_dir / 'training_loss_all_folds.png', show=show)

    plt.figure()
    for fold in range(len(all_val_losses)):
        plt.plot(all_val_losses[fold], line_styles[fold % len(line_styles)], label=f'Validation Loss Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss for All Folds')
    plt.legend()
    _finish(output_dir / 'validation_loss_all_folds.png', show=show)

    plt.figure()
    for fold in range(len(all_val_psnrs)):
        plt.plot(all_val_psnrs[fold], line_styles[fold % len(line_styles)], label=f'Validation PSNR Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Validation PSNR for All Folds')
    plt.legend()
    _finish(output_dir / 'validation_psnr_all_folds.png', show=show)

    min_epochs = min(len(x) for x in all_train_losses)
    avg_train_loss_all_folds = [sum([all_train_losses[fold][epoch] for fold in range(len(all_train_losses))]) / len(all_train_losses) for epoch in range(min_epochs)]
    avg_val_loss_all_folds = [sum([all_val_losses[fold][epoch] for fold in range(len(all_val_losses))]) / len(all_val_losses) for epoch in range(min_epochs)]
    avg_val_psnr_all_folds = [sum([all_val_psnrs[fold][epoch] for fold in range(len(all_val_psnrs))]) / len(all_val_psnrs) for epoch in range(min_epochs)]

    plt.figure()
    plt.plot(avg_train_loss_all_folds, label='Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Average Training Loss (MSE) Across All Folds')
    plt.legend()
    _finish(output_dir / 'avg_training_loss.png', show=show)

    plt.figure()
    plt.plot(avg_val_loss_all_folds, label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Average Validation Loss (MSE) Across All Folds')
    plt.legend()
    _finish(output_dir / 'avg_validation_loss.png', show=show)

    plt.figure()
    plt.plot(avg_val_psnr_all_folds, label='Average Validation PSNR Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Average Validation PSNR Across All Folds')
    plt.legend()
    _finish(output_dir / 'avg_validation_psnr_all_folds.png', show=show)

    plt.figure()
    plt.plot(avg_train_loss_all_folds, label='Training MSE', linestyle='--')
    plt.plot(avg_val_loss_all_folds, label='Validation MSE', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation MSE')
    plt.legend()
    _finish(output_dir / 'training_vs_validation_mse.png', show=show)

    # Two-panel figures retained for compact reporting
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    for fold in range(len(all_val_losses)):
        plt.plot(range(1, len(all_val_losses[fold]) + 1), all_val_losses[fold], label=f'Fold {fold + 1}')
    plt.title('Validation Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    for fold in range(len(all_val_psnrs)):
        plt.plot(range(1, len(all_val_psnrs[fold]) + 1), all_val_psnrs[fold], label=f'Fold {fold + 1}')
    plt.title('Validation PSNR Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    _finish(output_dir / 'validation_loss_psnr_across_folds.png', show=show)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(avg_val_loss_all_folds) + 1), avg_val_loss_all_folds, label='Validation Loss')
    plt.title('Average Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(avg_val_psnr_all_folds) + 1), avg_val_psnr_all_folds, label='Validation PSNR')
    plt.title('Average PSNR Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    _finish(output_dir / 'average_loss_psnr_across_folds.png', show=show)


def plot_psnr_distribution(psnr_distributions, output_path: str | Path, show: bool = False) -> None:
    plt.figure()
    plt.boxplot(psnr_distributions, labels=[f'Fold {i+1}' for i in range(len(psnr_distributions))])
    plt.xlabel('Fold')
    plt.ylabel('PSNR')
    plt.title('PSNR Distribution for Validation Data of Last Epoch in Main Training of Folds')
    _finish(output_path, show=show)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(tensor).save(path)
