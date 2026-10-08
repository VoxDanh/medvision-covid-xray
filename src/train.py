# ============================================================
# src/train.py
# TRAINING ONLY
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.factory import create_model
from src.dataset import create_dataloaders


# ============================================================
# UTILITIES
# ============================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# MAIN
# ============================================================

def run_training(
    model_config,
    scenario_config,
    experiment_id
):

    start_time = time.perf_counter()

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    base_config = load_yaml(
        "configs/base.yaml"
    )

    model_config_data = load_yaml(
        model_config
    )

    scenario_config_data = load_yaml(
        scenario_config
    )

    # ========================================================
    # SEED
    # ========================================================

    seed = base_config[
        "project"
    ]["seed"]

    set_seed(seed)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_name = (
        model_config_data[
            "model"
        ]["name"]
    )

    print("=" * 70)
    print("MEDVISION-CXR TRAINING")
    print("=" * 70)

    print(f"Experiment : {experiment_id}")
    print(f"Model      : {model_name}")
    print(f"Device     : {device}")

    if torch.cuda.is_available():

        print(
            "GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model(
        model_config_data
    )

    model = model.to(device)

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"Parameters : "
        f"{total_params:,}"
    )

    print("=" * 70)

    # ========================================================
    # DATA
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights
    ) = create_dataloaders(
        base_config,
        scenario_config_data
    )

    # ========================================================
    # LOSS
    # ========================================================

    if class_weights is not None:

        class_weights = (
            class_weights.to(device)
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print(
            "Class weights:",
            class_weights.tolist()
        )

    else:

        criterion = nn.CrossEntropyLoss()

        print(
            "Class weights: None"
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer_config = (
        base_config[
            "training"
        ]["optimizer"]
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=optimizer_config[
            "learning_rate"
        ],

        weight_decay=optimizer_config[
            "weight_decay"
        ]
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler_config = (
        base_config[
            "training"
        ]["scheduler"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=scheduler_config[
            "factor"
        ],

        patience=scheduler_config[
            "patience"
        ]
    )

    # ========================================================
    # TRAINING CONFIG
    # ========================================================

    training_config = (
        base_config["training"]
    )

    max_epochs = training_config[
        "epochs"
    ]

    early_stopping_config = (
        training_config[
            "early_stopping"
        ]
    )

    early_stopping_patience = (
        early_stopping_config[
            "patience"
        ]
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_root = Path(
        base_config[
            "output"
        ]["root"]
    )

    experiment_dir = (
        output_root
        / experiment_id
        / model_name
    )

    training_dir = (
        experiment_dir
        / "training"
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # HISTORY
    # ========================================================

    history = []

    best_val_loss = float("inf")

    best_epoch = 0

    patience_counter = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    for epoch in range(max_epochs):

        epoch_start = time.perf_counter()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        epoch_history = {

            "epoch":
                epoch + 1,

            "train_loss":
                float(train_loss),

            "train_accuracy":
                float(train_accuracy),

            "val_loss":
                float(val_loss),

            "val_accuracy":
                float(val_accuracy),

            "learning_rate":
                float(learning_rate),

            "epoch_time_seconds":
                float(epoch_time)
        }

        history.append(
            epoch_history
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{max_epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {learning_rate:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch + 1

            patience_counter = 0

            torch.save(
                model.state_dict(),
                training_dir
                / "best_model.pth"
            )

            print(
                "  -> Best model saved"
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            early_stopping_config["enabled"]
            and
            patience_counter
            >= early_stopping_patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # TRAINING TIME
    # ========================================================

    total_time = (
        time.perf_counter()
        - start_time
    )

    hours = int(
        total_time // 3600
    )

    minutes = int(
        (total_time % 3600) // 60
    )

    seconds = int(
        total_time % 60
    )

    time_formatted = (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        training_dir
        / "final_model.pth"
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    save_json(
        history,
        training_dir
        / "history.json"
    )

    # ========================================================
    # TRAINING SUMMARY
    # ========================================================

    training_summary = {

        "experiment_id":
            experiment_id,

        "model":
            model_name,

        "scenario":
            scenario_config_data[
                "scenario"
            ]["name"],

        "seed":
            seed,

        "device":
            str(device),

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "epochs_configured":
            max_epochs,

        "epochs_completed":
            len(history),

        "best_epoch":
            best_epoch,

        "best_val_loss":
            float(best_val_loss),

        "batch_size":
            training_config[
                "batch_size"
            ],

        "optimizer":
            optimizer_config[
                "name"
            ],

        "learning_rate":
            optimizer_config[
                "learning_rate"
            ],

        "weight_decay":
            optimizer_config[
                "weight_decay"
            ],

        "scheduler":
            scheduler_config[
                "name"
            ],

        "early_stopping":
            early_stopping_config[
                "enabled"
            ],

        "training_time_seconds":
            float(total_time),

        "training_time":
            time_formatted
    }

    save_json(
        training_summary,
        training_dir
        / "training_summary.json"
    )

    # ========================================================
    # SAVE CONFIG SNAPSHOT
    # ========================================================

    config_snapshot = {

        "base":
            base_config,

        "model":
            model_config_data,

        "scenario":
            scenario_config_data
    }

    save_json(
        config_snapshot,
        training_dir
        / "training_config.json"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best epoch    : {best_epoch}"
    )

    print(
        f"Best val loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Training time : "
        f"{time_formatted}"
    )

    print(
        f"Output        : "
        f"{training_dir}"
    )

    print("=" * 70)

    return training_summary
