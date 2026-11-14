# ============================================================
# src/train.py
# TRAINING ONLY
#
# - Loads configs relative to repository root
# - Supports configurable layer-wise freezing
# - Trains using train split
# - Uses validation metrics for checkpoint selection / early stopping
# - Does NOT evaluate on test split
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# CONFIG / UTILITIES
# ============================================================

def resolve_project_path(path):
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
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def format_duration(seconds):
    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ============================================================
# LAYER-WISE FREEZING
# ============================================================

def apply_layer_freezing(model, model_config):
    """
    Freeze selected modules based on YAML configuration.

    Example:

    freezing:
      enabled: true
      freeze_layers:
        - features.conv0
        - features.denseblock1

    All parameters inside the selected modules will be frozen.
    """

    freezing_config = model_config.get("freezing", {})

    freezing_enabled = bool(
        freezing_config.get("enabled", False)
    )

    if not freezing_enabled:
        print("\nFreezing: Disabled")
        return []

    freeze_layers = freezing_config.get(
        "freeze_layers", []
    )

    if not isinstance(freeze_layers, list):
        raise TypeError(
            "freezing.freeze_layers must be a list."
        )

    if not freeze_layers:
        raise ValueError(
            "Freezing is enabled but freeze_layers is empty."
        )

    # Map actual module names to modules.
    module_map = dict(model.named_modules())

    frozen_modules = []
    frozen_module_names = []

    for layer_name in freeze_layers:

        if layer_name not in module_map:
            raise ValueError(
                f"Layer '{layer_name}' does not exist in "
                f"model '{model_config.get('name')}'.\n"
                f"Check model.named_modules() for valid names."
            )

        module = module_map[layer_name]

        # Freeze all parameters belonging to this module.
        for parameter in module.parameters():
            parameter.requires_grad = False

        frozen_modules.append(module)
        frozen_module_names.append(layer_name)

    print("\nFreezing configuration:")
    print("-" * 60)

    for layer_name in frozen_module_names:
        print(f"Frozen layer: {layer_name}")

    print("-" * 60)

    return frozen_modules


def keep_frozen_batchnorm_in_eval(model, frozen_modules):
    """
    Keep BatchNorm modules inside frozen modules in eval mode.

    This prevents their running statistics from changing
    during training.
    """

    frozen_batchnorms = []

    for frozen_module in frozen_modules:

        for module in frozen_module.modules():

            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm
            ):
                module.eval()
                frozen_batchnorms.append(module)

    return frozen_batchnorms


def set_frozen_batchnorm_eval(frozen_batchnorms):
    """
    model.train() changes all modules to training mode.
    This function restores frozen BatchNorm modules to eval mode.
    """

    for module in frozen_batchnorms:
        module.eval()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    frozen_batchnorms=None
):

    model.train()

    # Restore eval mode for frozen BatchNorm modules.
    if frozen_batchnorms:
        set_frozen_batchnorm_eval(frozen_batchnorms)

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

        correct += (
            predictions == labels
        ).sum().item()

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

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    if total == 0:
        raise ValueError("Validation DataLoader is empty.")

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        running_loss / total,
        correct / total,
        float(val_f1)
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def run_training(
    model_config,
    scenario_config,
    experiment_id
):

    run_started_at = (
        datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Load configurations
    # --------------------------------------------------------

    base_config = load_yaml("configs/base.yaml")

    model_config_data = load_yaml(model_config)

    scenario_config_data = load_yaml(scenario_config)

    model_settings = model_config_data["model"]

    seed = int(base_config["project"]["seed"])

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_name = model_settings["name"]

    print("=" * 70)
    print("MEDVISION-CXR TRAINING")
    print("=" * 70)

    print(f"Repository : {PROJECT_ROOT}")
    print(f"Experiment : {experiment_id}")
    print(f"Model      : {model_name}")
    print(f"Device     : {device}")
    print(f"Started at : {run_started_at}")

    if torch.cuda.is_available():
        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(model_config_data)

    # Apply freezing before moving model to device.
    frozen_modules = apply_layer_freezing(
        model,
        model_settings
    )

    model = model.to(device)

    frozen_batchnorms = keep_frozen_batchnorm_in_eval(
        model,
        frozen_modules
    )

    total_params, trainable_params = count_parameters(model)

    frozen_params = total_params - trainable_params

    trainable_ratio = (
        trainable_params / total_params * 100
        if total_params > 0
        else 0
    )

    print("\nModel parameters:")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Frozen parameters    : {frozen_params:,}")
    print(f"Trainable ratio      : {trainable_ratio:.2f}%")

    if trainable_params == 0:
        raise ValueError(
            "No trainable parameters found. "
            "Check freezing configuration."
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, _test_loader, class_weights = (
        create_dataloaders(
            base_config,
            scenario_config_data
        )
    )

    print(f"\nTrain samples      : {len(train_loader.dataset):,}")
    print(f"Validation samples : {len(val_loader.dataset):,}")
    print(f"Train batches      : {len(train_loader):,}")
    print(f"Val batches        : {len(val_loader):,}")

    if class_weights is not None:

        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print(
            "Class weights : "
            f"{class_weights.detach().cpu().tolist()}"
        )

    else:

        criterion = nn.CrossEntropyLoss()

        print("Class weights : None")

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    training_config = base_config["training"]

    optimizer_config = training_config["optimizer"]

    scheduler_config = training_config["scheduler"]

    # IMPORTANT:
    # Only train parameters whose requires_grad is True.

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer_name = optimizer_config.get(
        "name", "AdamW"
    ).lower()

    if optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )

    elif optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            trainable_parameters,
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

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

    epochs = int(training_config["epochs"])

    early_stopping_config = training_config["early_stopping"]

    early_stopping_enabled = bool(
        early_stopping_config.get("enabled", True)
    )

    early_stopping_patience = int(
        early_stopping_config.get("patience", 10)
    )

    evaluation_config = base_config.get(
        "evaluation", {}
    )

    monitor = evaluation_config.get(
        "monitor", "f1"
    ).lower()

    monitor_mode = evaluation_config.get(
        "monitor_mode", "max"
    ).lower()

    if monitor not in {"f1", "loss", "accuracy"}:

        raise ValueError(
            f"Unsupported evaluation.monitor '{monitor}'. "
            "Supported values: f1, loss, accuracy."
        )

    if monitor_mode not in {"min", "max"}:

        raise ValueError(
            "evaluation.monitor_mode must be 'min' or 'max'."
        )

    if monitor == "loss" and monitor_mode != "min":

        raise ValueError(
            "monitor_mode must be 'min' when monitor='loss'."
        )

    if monitor in {"f1", "accuracy"} and monitor_mode != "max":

        raise ValueError(
            f"monitor_mode must be 'max' when monitor='{monitor}'."
        )

    print(f"\nCheckpoint monitor : {monitor} ({monitor_mode})")

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_root = resolve_project_path(
        base_config["output"]["root"]
    )

    training_dir = (
        output_root
        / experiment_id
        / model_name
        / "training"
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []

    best_metric = (
        float("-inf")
        if monitor_mode == "max"
        else float("inf")
    )

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
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            frozen_batchnorms,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        train_time = time.perf_counter() - train_start

        val_start = time.perf_counter()

        val_loss, val_accuracy, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        validation_time = time.perf_counter() - val_start

        scheduler.step(val_loss)

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_time = time.perf_counter() - epoch_start

        elapsed_training_time = (
            time.perf_counter() - training_loop_start
        )

        average_epoch_time = (
            elapsed_training_time / (epoch + 1)
        )

        eta_seconds = (
            average_epoch_time * (epochs - epoch - 1)
        )

        train_samples_per_second = (
            len(train_loader.dataset) / train_time
            if train_time > 0
            else None
        )

        if device.type == "cuda":

            gpu_allocated_mb = (
                torch.cuda.memory_allocated(device)
                / (1024 ** 2)
            )

            gpu_reserved_mb = (
                torch.cuda.memory_reserved(device)
                / (1024 ** 2)
            )

            gpu_peak_memory_mb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2)
            )

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
            "elapsed_training_time_seconds": float(
                elapsed_training_time
            ),
            "eta_seconds": float(eta_seconds),
            "train_samples_per_second": (
                float(train_samples_per_second)
                if train_samples_per_second is not None
                else None
            ),
            "gpu_memory_allocated_mb": (
                float(gpu_allocated_mb)
                if gpu_allocated_mb is not None
                else None
            ),
            "gpu_memory_reserved_mb": (
                float(gpu_reserved_mb)
                if gpu_reserved_mb is not None
                else None
            ),
            "gpu_peak_memory_mb": (
                float(gpu_peak_memory_mb)
                if gpu_peak_memory_mb is not None
                else None
            ),
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
                f"  GPU memory: "
                f"allocated={gpu_allocated_mb:.0f} MB, "
                f"reserved={gpu_reserved_mb:.0f} MB, "
                f"peak={gpu_peak_memory_mb:.0f} MB"
            )

        # ----------------------------------------------------
        # Checkpoint selection
        # ----------------------------------------------------

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

            torch.save(
                model.state_dict(),
                training_dir / "best_model.pth"
            )

            print("  -> Best model saved")

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        if base_config["output"].get("save_history", True):

            save_json(
                history,
                training_dir / "history.json"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            early_stopping_enabled
            and patience_counter >= early_stopping_patience
        ):

            print(
                f"\nEarly stopping at epoch {epoch + 1}."
            )

            break

    if best_epoch == 0:

        raise RuntimeError(
            "Training ended without saving a best checkpoint."
        )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    if base_config["output"].get("save_final_model", True):

        torch.save(
            model.state_dict(),
            training_dir / "final_model.pth"
        )

    if base_config["output"].get("save_history", True):

        save_json(
            history,
            training_dir / "history.json"
        )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    training_loop_time = (
        time.perf_counter() - training_loop_start
    )

    total_time = time.perf_counter() - start_time

    run_finished_at = (
        datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )

    time_formatted = format_duration(total_time)

    training_loop_time_formatted = format_duration(
        training_loop_time
    )

    average_epoch_time_final = (
        training_loop_time / len(history)
        if history
        else 0.0
    )

    peak_gpu_memory_mb_final = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    freezing_config = model_settings.get(
        "freezing", {}
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
        "frozen_parameters": int(frozen_params),
        "trainable_ratio_percent": float(trainable_ratio),

        "freezing_enabled": bool(
            freezing_config.get("enabled", False)
        ),

        "freeze_layers": freezing_config.get(
            "freeze_layers", []
        ),

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

        "total_runtime_seconds_including_setup_and_saving": float(
            total_time
        ),

        "total_runtime": time_formatted,

        "average_epoch_time_seconds": float(
            average_epoch_time_final
        ),

        "peak_gpu_memory_mb": (
            float(peak_gpu_memory_mb_final)
            if peak_gpu_memory_mb_final is not None
            else None
        ),

        "train_samples": int(len(train_loader.dataset)),
        "validation_samples": int(len(val_loader.dataset)),

        "train_batches_per_epoch": int(len(train_loader)),
        "validation_batches": int(len(val_loader)),
    }

    save_json(
        training_summary,
        training_dir / "training_summary.json"
    )

    # --------------------------------------------------------
    # Save configuration snapshot
    # --------------------------------------------------------

    config_snapshot = {
        "base": base_config,
        "model": model_config_data,
        "scenario": scenario_config_data,
    }

    save_json(
        config_snapshot,
        training_dir / "training_config.json"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Model             : {model_name}")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best {monitor:>5}       : {best_metric:.4f}")

    print(f"Total parameters  : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")
    print(f"Frozen params     : {frozen_params:,}")

    print(f"Training-loop time: {training_loop_time_formatted}")
    print(f"Average epoch time: {average_epoch_time_final:.1f}s")

    if peak_gpu_memory_mb_final is not None:

        print(
            f"Peak GPU memory   : "
            f"{peak_gpu_memory_mb_final:.0f} MB"
        )

    print(f"Total runtime     : {time_formatted}")
    print(f"Finished at       : {run_finished_at}")
    print(f"Output directory  : {training_dir}")

    print("=" * 70)

    return training_summary
