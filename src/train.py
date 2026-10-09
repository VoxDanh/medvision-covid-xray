# ============================================================
# src/train.py
# TRAINING ONLY
# - Loads configs relative to the repository root
# - Trains using train split
# - Uses validation metrics for checkpoint selection / early stopping
# - Does NOT evaluate on the test split
# ============================================================

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from src.models.factory import create_model
from src.dataset import create_dataloaders


# Repository root: .../medvision-covid-xray
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# CONFIG / UTILITIES
# ============================================================

def resolve_project_path(path):
    """Resolve an absolute path as-is, or a relative path from PROJECT_ROOT."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path):
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"YAML config not found: {path}\n"
            f"Repository root: {PROJECT_ROOT}"
        )
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        raise ValueError(f"YAML config is empty: {path}")
    return data


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def format_duration(seconds):
    """Format seconds as HH:MM:SS (hours may exceed 24)."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Train DataLoader is empty.")

    return running_loss / total, correct / total


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    if total == 0:
        raise ValueError("Validation DataLoader is empty.")

    # Macro F1 avoids assuming which numeric label corresponds to COVID.
    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return running_loss / total, correct / total, float(val_f1)


# ============================================================
# MAIN TRAINING
# ============================================================

def run_training(model_config, scenario_config, experiment_id):
    run_started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Load configuration files robustly
    # --------------------------------------------------------
    base_config = load_yaml("configs/base.yaml")
    model_config_data = load_yaml(model_config)
    scenario_config_data = load_yaml(scenario_config)

    seed = int(base_config["project"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_config_data["model"]["name"]

    print("=" * 70)
    print("MEDVISION-CXR TRAINING")
    print("=" * 70)
    print(f"Repository : {PROJECT_ROOT}")
    print(f"Experiment : {experiment_id}")
    print(f"Model      : {model_name}")
    print(f"Device     : {device}")
    print(f"Started at : {run_started_at}")

    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = create_model(model_config_data).to(device)
    total_params, trainable_params = count_parameters(model)

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # --------------------------------------------------------
    # Data
    # The dataset module returns train, validation, test, weights.
    # The test loader is intentionally not used in training.
    # --------------------------------------------------------
    train_loader, val_loader, _test_loader, class_weights = (
        create_dataloaders(base_config, scenario_config_data)
    )

    print(f"Train samples      : {len(train_loader.dataset):,}")
    print(f"Validation samples : {len(val_loader.dataset):,}")
    print(f"Train batches      : {len(train_loader):,}")
    print(f"Val batches        : {len(val_loader):,}")

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights : {class_weights.detach().cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Class weights : None")

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------
    training_config = base_config["training"]
    optimizer_config = training_config["optimizer"]
    scheduler_config = training_config["scheduler"]

    optimizer_name = optimizer_config.get("name", "AdamW").lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_config.get('name')}'. "
            "Add its implementation in train.py."
        )

    scheduler_name = scheduler_config.get(
        "name", "ReduceLROnPlateau"
    ).lower()
    if scheduler_name == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
        )
    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_config.get('name')}'. "
            "Add its implementation in train.py."
        )

    epochs = int(training_config["epochs"])
    early_stopping_config = training_config["early_stopping"]
    early_stopping_enabled = bool(early_stopping_config.get("enabled", True))
    early_stopping_patience = int(early_stopping_config.get("patience", 10))

    evaluation_config = base_config.get("evaluation", {})
    monitor = evaluation_config.get("monitor", "f1").lower()
    monitor_mode = evaluation_config.get("monitor_mode", "max").lower()

    if monitor not in {"f1", "loss", "accuracy"}:
        raise ValueError(
            f"Unsupported evaluation.monitor '{monitor}'. "
            "Supported values: f1, loss, accuracy."
        )
    if monitor_mode not in {"min", "max"}:
        raise ValueError("evaluation.monitor_mode must be 'min' or 'max'.")

    # F1 and accuracy should be maximized; loss should be minimized.
    if monitor == "loss" and monitor_mode != "min":
        raise ValueError("monitor_mode must be 'min' when monitor='loss'.")
    if monitor in {"f1", "accuracy"} and monitor_mode != "max":
        raise ValueError(
            f"monitor_mode must be 'max' when monitor='{monitor}'."
        )

    print(f"Checkpoint monitor : {monitor} ({monitor_mode})")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------
    output_root = resolve_project_path(base_config["output"]["root"])
    training_dir = output_root / experiment_id / model_name / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    patience_counter = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)
    training_loop_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_time = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_loss, val_accuracy, val_f1 = validate(
            model, val_loader, criterion, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_time = time.perf_counter() - val_start

        # Keep scheduler behavior based on validation loss.
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - epoch_start
        elapsed_training_time = time.perf_counter() - training_loop_start
        average_epoch_time = elapsed_training_time / (epoch + 1)
        eta_seconds = average_epoch_time * (epochs - epoch - 1)
        train_samples_per_second = len(train_loader.dataset) / train_time if train_time > 0 else None

        if device.type == "cuda":
            gpu_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            gpu_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            gpu_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            gpu_allocated_mb = None
            gpu_reserved_mb = None
            gpu_peak_memory_mb = None

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "val_f1": float(val_f1),
            "learning_rate": current_lr,
            "train_time_seconds": float(train_time),
            "validation_time_seconds": float(validation_time),
            "epoch_time_seconds": float(epoch_time),
            "elapsed_training_time_seconds": float(elapsed_training_time),
            "eta_seconds": float(eta_seconds),
            "train_samples_per_second": float(train_samples_per_second) if train_samples_per_second is not None else None,
            "gpu_memory_allocated_mb": float(gpu_allocated_mb) if gpu_allocated_mb is not None else None,
            "gpu_memory_reserved_mb": float(gpu_reserved_mb) if gpu_reserved_mb is not None else None,
            "gpu_peak_memory_mb": float(gpu_peak_memory_mb) if gpu_peak_memory_mb is not None else None,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Train time: {train_time:.1f}s | "
            f"Val time: {validation_time:.1f}s | "
            f"Epoch: {format_duration(epoch_time)} | "
            f"Elapsed: {format_duration(elapsed_training_time)} | "
            f"ETA: {format_duration(eta_seconds)} | "
            f"Throughput: {train_samples_per_second:.1f} samples/s"
        )
        if gpu_peak_memory_mb is not None:
            print(
                f"  GPU memory: allocated={gpu_allocated_mb:.0f} MB, "
                f"reserved={gpu_reserved_mb:.0f} MB, "
                f"peak={gpu_peak_memory_mb:.0f} MB"
            )

        metric_values = {
            "f1": val_f1,
            "loss": val_loss,
            "accuracy": val_accuracy,
        }
        current_metric = metric_values[monitor]

        if monitor_mode == "max":
            is_best = current_metric > best_metric
        else:
            is_best = current_metric < best_metric

        if is_best:
            best_metric = float(current_metric)
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), training_dir / "best_model.pth")
            print("  -> Best model saved")
        else:
            patience_counter += 1

        # Save history each epoch so progress is not lost if runtime stops.
        if base_config["output"].get("save_history", True):
            save_json(history, training_dir / "history.json")

        if early_stopping_enabled and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    if best_epoch == 0:
        raise RuntimeError("Training ended without saving a best checkpoint.")

    # --------------------------------------------------------
    # Save final model and artifacts
    # --------------------------------------------------------
    if base_config["output"].get("save_final_model", True):
        torch.save(model.state_dict(), training_dir / "final_model.pth")

    if base_config["output"].get("save_history", True):
        save_json(history, training_dir / "history.json")

    training_loop_time = time.perf_counter() - training_loop_start
    total_time = time.perf_counter() - start_time
    run_finished_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    time_formatted = format_duration(total_time)
    training_loop_time_formatted = format_duration(training_loop_time)
    average_epoch_time_final = training_loop_time / len(history) if history else 0.0
    peak_gpu_memory_mb_final = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    training_summary = {
        "experiment_id": experiment_id,
        "model": model_name,
        "scenario": scenario_config_data["scenario"]["name"],
        "seed": seed,
        "device": str(device),
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "epochs_configured": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "best_metric": float(best_metric),
        "batch_size": int(training_config["batch_size"]),
        "optimizer": optimizer_config["name"],
        "learning_rate": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
        "scheduler": scheduler_config["name"],
        "early_stopping_enabled": early_stopping_enabled,
        "training_time_seconds": float(training_loop_time),
        "training_time": training_loop_time_formatted,
        "total_runtime_seconds_including_setup_and_saving": float(total_time),
        "total_runtime": time_formatted,
        "average_epoch_time_seconds": float(average_epoch_time_final),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb_final) if peak_gpu_memory_mb_final is not None else None,
        "train_samples": int(len(train_loader.dataset)),
        "validation_samples": int(len(val_loader.dataset)),
        "train_batches_per_epoch": int(len(train_loader)),
        "validation_batches": int(len(val_loader)),
    }
    save_json(training_summary, training_dir / "training_summary.json")

    config_snapshot = {
        "base": base_config,
        "model": model_config_data,
        "scenario": scenario_config_data,
    }
    save_json(config_snapshot, training_dir / "training_config.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model             : {model_name}")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best {monitor:>5}       : {best_metric:.4f}")
    print(f"Training-loop time: {training_loop_time_formatted}")
    print(f"Average epoch time: {average_epoch_time_final:.1f}s")
    if peak_gpu_memory_mb_final is not None:
        print(f"Peak GPU memory   : {peak_gpu_memory_mb_final:.0f} MB")
    print(f"Total runtime     : {time_formatted}")
    print(f"Finished at       : {run_finished_at}")
    print(f"Output directory  : {training_dir}")
    print("=" * 70)

    return training_summary
