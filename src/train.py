# ============================================================
# src/train.py
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.factory import create_model


# ============================================================
# CONFIG
# ============================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL PARAMETERS
# ============================================================

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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

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

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


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

    running_loss = 0.0

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

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# TEST PREDICTIONS
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    # --------------------------------------------------------
    # SPECIFICITY
    # --------------------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    auroc = roc_auc_score(
        labels,
        probabilities[:, 1]
    )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        labels,
        probabilities[:, 1]
    )

    return {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "specificity": float(
            specificity
        ),

        "f1": float(f1),

        "auroc": float(auroc),

        "pr_auc": float(pr_auc),

        "confusion_matrix": {

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    path
):

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


# ============================================================
# MAIN TRAINING
# ============================================================

def run_training(
    model_config,
    scenario_config,
    experiment_id
):

    # ========================================================
    # START TIMER
    # ========================================================

    total_start_time = time.perf_counter()

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

    print("=" * 70)
    print("MEDVISION-CXR TRAINING")
    print("=" * 70)

    print(
        f"Experiment : {experiment_id}"
    )

    print(
        f"Model      : "
        f"{model_config_data['model']['name']}"
    )

    print(
        f"Device     : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)

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
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    # TODO:
    #
    # dataset.py sẽ trả về:
    #
    # train_loader
    # val_loader
    # test_loader
    # class_weights
    #
    # Ví dụ:
    #
    # from src.dataset import create_dataloaders
    #
    # (
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     class_weights
    # ) = create_dataloaders(
    #     base_config,
    #     scenario_config_data
    # )

    from src.dataset import create_dataloaders

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
    # CLASS WEIGHT
    # ========================================================

    if class_weights is not None:

        class_weights = (
            class_weights.to(device)
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        print(
            f"Class weights : "
            f"{class_weights.tolist()}"
        )

    else:

        criterion = nn.CrossEntropyLoss()

        print(
            "Class weights : None"
        )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer_config = (
        base_config["training"]
        ["optimizer"]
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
        base_config["training"]
        ["scheduler"]
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

    epochs = training_config[
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
    # OUTPUT DIRECTORY
    # ========================================================

    output_root = Path(
        base_config[
            "output"
        ]["root"]
    )

    model_name = (
        model_config_data[
            "model"
        ]["name"]
    )

    experiment_dir = (
        output_root
        / experiment_id
        / model_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    history = []

    best_val_loss = float("inf")

    best_epoch = 0

    patience_counter = 0

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("START TRAINING")
    print("=" * 70)

    for epoch in range(epochs):

        epoch_start_time = (
            time.perf_counter()
        )

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

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # EPOCH TIME
        # ----------------------------------------------------

        epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        epoch_result = {

            "epoch": epoch + 1,

            "train_loss": float(
                train_loss
            ),

            "train_accuracy": float(
                train_accuracy
            ),

            "val_loss": float(
                val_loss
            ),

            "val_accuracy": float(
                val_accuracy
            ),

            "learning_rate": float(
                current_lr
            ),

            "epoch_time_seconds": float(
                epoch_time
            )
        }

        history.append(
            epoch_result
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.2e} | "
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
                experiment_dir
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
            early_stopping_config[
                "enabled"
            ]
            and
            patience_counter
            >= early_stopping_patience
        ):

            print(
                f"\nEarly stopping "
                f"at epoch {epoch + 1}"
            )

            break

    # ========================================================
    # TOTAL TRAINING TIME
    # ========================================================

    total_training_time = (
        time.perf_counter()
        - total_start_time
    )

    hours = int(
        total_training_time // 3600
    )

    minutes = int(
        (total_training_time % 3600)
        // 60
    )

    seconds = int(
        total_training_time % 60
    )

    training_time_formatted = (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    if base_config[
        "output"
    ]["save_final_model"]:

        torch.save(
            model.state_dict(),
            experiment_dir
            / "final_model.pth"
        )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    if base_config[
        "output"
    ]["save_history"]:

        save_json(
            history,
            experiment_dir
            / "history.json"
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    best_model_path = (
        experiment_dir
        / "best_model.pth"
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TESTING BEST MODEL")
    print("=" * 70)

    test_labels, test_predictions, test_probabilities = (
        predict(
            model,
            test_loader,
            device
        )
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_predictions,
        test_probabilities
    )

    # ========================================================
    # DISPLAY TEST RESULTS
    # ========================================================

    print(
        f"Accuracy    : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Specificity : "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"F1          : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"AUROC       : "
        f"{test_metrics['auroc']:.4f}"
    )

    print(
        f"PR-AUC      : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_metrics[
            "confusion_matrix"
        ]
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    if base_config[
        "output"
    ]["save_predictions"]:

        prediction_data = {

            "labels": test_labels.tolist(),

            "predictions":
                test_predictions.tolist(),

            "probability_normal":
                test_probabilities[:, 0].tolist(),

            "probability_covid":
                test_probabilities[:, 1].tolist()
        }

        save_json(
            prediction_data,
            experiment_dir
            / "predictions.json"
        )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = {

        "experiment": {

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
                str(device)
        },

        "model_info": {

            "total_parameters":
                total_params,

            "trainable_parameters":
                trainable_params
        },

        "training": {

            "epochs_configured":
                epochs,

            "epochs_completed":
                len(history),

            "best_epoch":
                best_epoch,

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
                total_training_time,

            "training_time":
                training_time_formatted
        },

        "test_metrics":
            test_metrics,

        "files": {

            "best_model":
                str(
                    experiment_dir
                    / "best_model.pth"
                ),

            "final_model":
                str(
                    experiment_dir
                    / "final_model.pth"
                ),

            "history":
                str(
                    experiment_dir
                    / "history.json"
                ),

            "predictions":
                str(
                    experiment_dir
                    / "predictions.json"
                )
        }
    }

    save_json(
        result,
        experiment_dir
        / "result.json"
    )

    # ========================================================
    # SAVE CONFIG
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
        experiment_dir
        / "config.json"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Model              : {model_name}"
    )

    print(
        f"Best epoch         : {best_epoch}"
    )

    print(
        f"Test F1            : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test AUROC         : "
        f"{test_metrics['auroc']:.4f}"
    )

    print(
        f"Test PR-AUC        : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Training time      : "
        f"{training_time_formatted}"
    )

    print(
        f"Output directory   : "
        f"{experiment_dir}"
    )

    print("=" * 70)

    return result
