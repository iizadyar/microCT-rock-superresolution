from __future__ import annotations

import logging
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split

from .config import output_root
from .data import PairedImageDataset
from .losses import build_loss
from .metrics import mse_value, psnr_from_mse, psnr_value
from .models import build_model, model_uses_preupsampling
from .preprocessing import bicubic_resize
from .results import write_csv, write_json
from .plotting import (
    plot_optimization,
    plot_best_trial_curves,
    plot_training_validation_curves,
    plot_psnr_distribution,
    save_tensor_image,
)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    requested = cfg.get("project", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def use_plots(cfg: dict) -> bool:
    return bool(cfg.get("outputs", {}).get("save_plots", True))


def show_plots(cfg: dict) -> bool:
    return bool(cfg.get("outputs", {}).get("show_plots", False))


def save_images_enabled(cfg: dict) -> bool:
    return bool(cfg.get("outputs", {}).get("save_images", True))


def setup_logger(project_dir: Path) -> logging.Logger:
    project_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(project_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(project_dir / "training.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_print(logger: logging.Logger, message: str) -> None:
    print(message)
    logger.info(message)


def use_augmentation(cfg: dict, split: str) -> bool:
    aug = cfg.get("augmentation", {})
    if not aug.get("enabled", True):
        return False
    if split == "train":
        return bool(aug.get("apply_to_train", True))
    return bool(aug.get("apply_to_validation", True))


def make_dataset(cfg: dict, model_name: str, split: str, indices=None):
    d = cfg["dataset"]
    aug = cfg.get("augmentation", {})
    dataset = PairedImageDataset(
        lr_dir=d["lr_dir"],
        hr_dir=d["hr_dir"],
        scale=int(d["scale"]),
        augment=use_augmentation(cfg, split),
        deterministic_augment=(split != "train"),
        pre_upscale=model_uses_preupsampling(model_name),
        crop_to_scale=bool(d.get("crop_to_scale", True)),
        horizontal_flip=bool(aug.get("horizontal_flip", True)),
        vertical_flip=bool(aug.get("vertical_flip", True)),
        rot90=bool(aug.get("rot90", True)),
    )
    if indices is not None:
        return Subset(dataset, list(indices))
    return dataset


def make_loader(dataset, batch_size: int, shuffle: bool, cfg: dict):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.get("project", {}).get("num_workers", 0)),
    )


def match_output(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if output.shape[-2:] != target.shape[-2:]:
        output = F.interpolate(output, size=target.shape[-2:], mode="bicubic", align_corners=False)
    return output


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for lr_image, hr_image, _ in train_loader:
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        optimizer.zero_grad()
        outputs = match_output(model(lr_image), hr_image)
        loss = criterion(outputs, hr_image)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(train_loader), 1)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_losses_epoch = []
    val_psnr_epoch = []
    rows = []
    with torch.no_grad():
        for lr_image, hr_image, stems in val_loader:
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            outputs = match_output(model(lr_image), hr_image).clamp(0, 1)
            val_loss = criterion(outputs, hr_image).item()
            val_losses_epoch.append(val_loss)
            for i in range(len(outputs)):
                mse = mse_value(outputs[i], hr_image[i])
                psnr = psnr_from_mse(mse)
                val_psnr_epoch.append(psnr)
                rows.append({"image": stems[i], "mse": mse, "psnr": psnr})
    avg_val_loss = sum(val_losses_epoch) / max(len(val_losses_epoch), 1)
    avg_psnr = sum(val_psnr_epoch) / max(len(val_psnr_epoch), 1)
    avg_mse = sum([r["mse"] for r in rows]) / max(len(rows), 1)
    return avg_val_loss, avg_mse, avg_psnr, rows


def compute_psnr_values(model, val_loader, device):
    model.eval()
    psnr_values = []
    with torch.no_grad():
        for lr_image, hr_image, _ in val_loader:
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            outputs = match_output(model(lr_image), hr_image).clamp(0, 1)
            for i in range(lr_image.size(0)):
                psnr = psnr_value(outputs[i], hr_image[i])
                psnr_values.append(psnr)
    return psnr_values


def validate_and_save(model, val_loader, device, output_dir: Path, max_images: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch_idx, (lr_image, hr_image, stems) in enumerate(val_loader):
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            outputs = match_output(model(lr_image), hr_image).clamp(0, 1)
            for i in range(lr_image.size(0)):
                if max_images > 0 and saved >= max_images:
                    return
                stem = str(stems[i])
                save_tensor_image(lr_image[i].cpu(), output_dir / f"lr_image_{stem}.png")
                save_tensor_image(hr_image[i].cpu(), output_dir / f"hr_image_{stem}.png")
                save_tensor_image(outputs[i].cpu(), output_dir / f"super_res_image_{stem}.png")
                bicubic = bicubic_resize(lr_image[i].cpu(), size=hr_image[i].shape[-2:]).clamp(0, 1)
                save_tensor_image(bicubic, output_dir / f"bicubic_image_{stem}.png")
                saved += 1


def run_hpo(cfg: dict, model_name: str, all_indices: list[int], device, project_dir: Path, logger: logging.Logger):
    hpo_cfg = cfg.get("hpo", {})
    if not hpo_cfg.get("enabled", True):
        return {
            "lr": float(cfg["training"].get("learning_rate", 1e-4)),
            "batch_size": int(cfg["training"].get("batch_size", 16)),
            "optimization_time": 0.0,
            "study": None,
        }

    import optuna

    optimization_start_time = time.time()
    if len(all_indices) < 3:
        return {
            "lr": float(cfg["training"].get("learning_rate", 1e-4)),
            "batch_size": int(cfg["training"].get("batch_size", 16)),
            "optimization_time": 0.0,
            "study": None,
        }

    hpo_indices, _ = train_test_split(
        all_indices,
        train_size=float(hpo_cfg.get("split_fraction_for_hpo", 0.5)),
        random_state=int(cfg.get("project", {}).get("seed", 42)),
    )
    if len(hpo_indices) < 3:
        hpo_indices = all_indices

    train_indices, val_indices = train_test_split(hpo_indices, test_size=0.2, random_state=42)

    def objective(trial):
        lr = trial.suggest_float("lr", float(hpo_cfg.get("lr_min", 1e-5)), float(hpo_cfg.get("lr_max", 1e-3)), log=True)
        batch_size = trial.suggest_categorical("batch_size", [int(x) for x in hpo_cfg.get("batch_sizes", [8, 16, 32])])
        train_dataset = make_dataset(cfg, model_name, "train", train_indices)
        val_dataset = make_dataset(cfg, model_name, "validation", val_indices)
        log_print(logger, f"Number of training samples: {len(train_dataset)}")
        log_print(logger, f"Number of validation samples: {len(val_dataset)}")
        train_loader = make_loader(train_dataset, batch_size, True, cfg)
        val_loader = make_loader(val_dataset, batch_size, False, cfg)
        model = build_model(model_name, int(cfg["dataset"]["scale"]), cfg.get("models", {}).get(model_name, {})).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = build_loss(cfg["training"].get("loss", "mse")).to(device)
        epochs = int(hpo_cfg.get("epochs", 5))
        train_losses, val_losses, val_psnrs = [], [], []
        avg_psnr = 0.0
        for epoch in range(epochs):
            log_print(logger, f"Epoch {epoch + 1}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            log_print(logger, f"Training Loss: {train_loss:.6f}")
            val_loss, _, avg_psnr, _ = evaluate_model(model, val_loader, criterion, device)
            log_print(logger, f"Validation Loss: {val_loss:.6f}")
            log_print(logger, f"Average PSNR on Validation: {avg_psnr:.6f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_psnrs.append(avg_psnr)
        trial.set_user_attr("train_losses", train_losses)
        trial.set_user_attr("val_losses", val_losses)
        trial.set_user_attr("val_psnrs", val_psnrs)
        return avg_psnr

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(hpo_cfg.get("n_trials", 2)))
    optimization_end_time = time.time()
    log_print(logger, f"Best hyperparameters: {study.best_params}")

    if use_plots(cfg):
        plot_optimization(study, project_dir / "optimization_history.png", show=show_plots(cfg))
        plot_best_trial_curves(study.best_trial, project_dir, show=show_plots(cfg))

    return {
        "lr": float(study.best_params["lr"]),
        "batch_size": int(study.best_params["batch_size"]),
        "optimization_time": optimization_end_time - optimization_start_time,
        "study": study,
    }


def train_models(cfg: dict, models: list[str]):
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    start_time = time.time()
    device = get_device(cfg)
    dataset_name = cfg["dataset"]["name"]
    scale = int(cfg["dataset"]["scale"])
    out_root = output_root(cfg) / f"{dataset_name.lower()}_{scale}x"
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name in models:
        project_dir = out_root / model_name
        project_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(project_dir)
        log_print(logger, f"Running model: {model_name}")

        base_dataset = make_dataset(cfg, model_name, "validation")
        all_indices = list(range(len(base_dataset)))

        hpo_result = run_hpo(cfg, model_name, all_indices, device, project_dir, logger)
        optimization_time = hpo_result["optimization_time"]
        best_lr = hpo_result["lr"]
        best_batch_size = hpo_result["batch_size"]
        study = hpo_result["study"]
        write_json(project_dir / "best_hyperparameters.json", {"lr": best_lr, "batch_size": best_batch_size})
        if study is not None:
            trials_rows = study.trials_dataframe().to_dict(orient="records")
            write_csv(project_dir / "hpo_trials.csv", trials_rows)

        training_start_time = time.time()
        if cfg.get("hpo", {}).get("enabled", True) and len(all_indices) >= 10:
            _, cv_indices = train_test_split(
                all_indices,
                test_size=1.0 - float(cfg.get("hpo", {}).get("split_fraction_for_hpo", 0.5)),
                random_state=int(cfg.get("project", {}).get("seed", 42)),
            )
        else:
            cv_indices = all_indices

        k_folds = min(int(cfg["training"].get("k_folds", 5)), len(cv_indices))
        k_folds = max(2, k_folds)
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_train_losses, all_val_losses, all_val_psnrs = [], [], []
        fold_summary_rows, per_image_rows = [], []
        psnr_distributions = []
        criterion = build_loss(cfg["training"].get("loss", "mse")).to(device)

        avg_val_psnr = 0.0
        for fold, (train_rel, val_rel) in enumerate(kfold.split(cv_indices)):
            train_indices = [cv_indices[i] for i in train_rel]
            val_indices = [cv_indices[i] for i in val_rel]
            train_dataset = make_dataset(cfg, model_name, "train", train_indices)
            val_dataset = make_dataset(cfg, model_name, "validation", val_indices)
            log_print(logger, f"Fold {fold + 1}")
            log_print(logger, f"Number of training samples: {len(train_dataset)}")
            log_print(logger, f"Number of validation samples: {len(val_dataset)}")
            train_loader = make_loader(train_dataset, best_batch_size, True, cfg)
            val_loader = make_loader(val_dataset, best_batch_size, False, cfg)
            model = build_model(model_name, scale, cfg.get("models", {}).get(model_name, {})).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

            train_losses, val_losses, val_psnrs = [], [], []
            best_val_loss = float("inf")
            patience_counter = 0
            early_stopping_patience = int(cfg["training"].get("early_stopping_patience", 0))
            epochs = int(cfg["training"].get("epochs", 150))
            rows = []

            for epoch in range(epochs):
                log_print(logger, f"Epoch {epoch + 1}")
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                log_print(logger, f"Training Loss: {train_loss:.6f}")
                val_loss, val_mse, avg_psnr, rows = evaluate_model(model, val_loader, criterion, device)
                log_print(logger, f"Validation Loss: {val_loss:.6f}")
                log_print(logger, f"Average PSNR on Validation: {avg_psnr:.6f}")
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_psnrs.append(avg_psnr)

                if early_stopping_patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        log_print(logger, f"Early stopping at epoch {epoch + 1}")
                        break

            psnr_values_fold = compute_psnr_values(model, val_loader, device)
            log_print(logger, f"PSNR values for Fold {fold + 1}:")
            log_print(logger, str(psnr_values_fold))
            psnr_distributions.append(psnr_values_fold)

            if save_images_enabled(cfg):
                max_images = int(cfg["training"].get("save_images_per_fold", 0))
                output_dir = project_dir / f"output_images_fold_{fold}"
                validate_and_save(model, val_loader, device, output_dir, max_images=max_images)
                log_print(logger, "Super-resolved images saved successfully.")

            for r in rows:
                r["fold"] = fold + 1
                r["model"] = model_name
                per_image_rows.append(r)

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_val_psnrs.append(val_psnrs)
            avg_val_psnr += avg_psnr / k_folds
            fold_summary_rows.append({
                "fold": fold + 1,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "final_val_psnr": val_psnrs[-1],
                "epochs_completed": len(val_psnrs),
            })

        avg_val_loss = sum(row["final_val_loss"] for row in fold_summary_rows) / len(fold_summary_rows)
        training_end_time = time.time()
        final_time = training_end_time - start_time
        training_time = training_end_time - training_start_time
        log_print(logger, f"Final time (total time): {final_time} seconds")
        log_print(logger, f"Optimization time: {optimization_time} seconds")
        log_print(logger, f"Final training time: {training_time} seconds")
        log_print(logger, f"Final average PSNR: {avg_val_psnr:.6f}")

        if use_plots(cfg):
            plot_training_validation_curves(all_train_losses, all_val_losses, all_val_psnrs, project_dir, show=show_plots(cfg))
            plot_psnr_distribution(psnr_distributions, project_dir / "psnr_distribution_boxplot.png", show=show_plots(cfg))

        write_csv(project_dir / "fold_summary.csv", fold_summary_rows)
        write_csv(project_dir / "per_image_metrics.csv", per_image_rows)
        summary = {
            "dataset": dataset_name,
            "scale": scale,
            "model": model_name,
            "learning_rate": best_lr,
            "batch_size": best_batch_size,
            "average_validation_loss": avg_val_loss,
            "average_validation_psnr": avg_val_psnr,
            "final_time_seconds": final_time,
            "optimization_time_seconds": optimization_time,
            "training_time_seconds": training_time,
        }
        write_csv(project_dir / "summary.csv", [summary])
        summaries.append(summary)

    write_csv(out_root / "model_summary.csv", summaries)
    return summaries


def evaluate_bicubic(cfg: dict):
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    dataset_name = cfg["dataset"]["name"]
    scale = int(cfg["dataset"]["scale"])
    out_dir = output_root(cfg) / f"{dataset_name.lower()}_{scale}x" / "bicubic"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    dataset = make_dataset(cfg, "bicubic", "validation")
    rows = []
    start = time.time()
    for lr, hr, stem in dataset:
        bicubic = bicubic_resize(lr, size=hr.shape[-2:]).clamp(0, 1)
        mse = mse_value(bicubic, hr)
        psnr = psnr_from_mse(mse)
        rows.append({"image": stem, "mse": mse, "psnr": psnr})
    runtime = time.time() - start
    avg_mse = sum(r["mse"] for r in rows) / max(len(rows), 1)
    avg_psnr = sum(r["psnr"] for r in rows) / max(len(rows), 1)
    log_print(logger, f"Final time (total time): {runtime} seconds")
    log_print(logger, f"Validation Loss: {avg_mse:.6f}")
    log_print(logger, f"Average PSNR on Validation: {avg_psnr:.6f}")
    write_csv(out_dir / "per_image_metrics.csv", rows)
    summary = {"dataset": dataset_name, "scale": scale, "model": "bicubic", "average_validation_mse": avg_mse, "average_validation_psnr": avg_psnr, "runtime_seconds": runtime}
    write_csv(out_dir / "summary.csv", [summary])
    return summary
