# ============================================================
# src/evaluation.py
# TEST / EVALUATION ONLY
# ============================================================

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

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


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device
):

    model.eval()

    labels = []
    predictions = []
    probabilities = []

    for images, batch_labels in loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = probs.argmax(
            dim=1
        )

        labels.extend(
            batch_labels.numpy()
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )

    return (
        np.array(labels),
        np.array(predictions),
        np.array(probabilities)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    auroc = roc_auc_score(
        labels,
        probabilities[:, 1]
    )

    pr_auc = average_precision_score(
        labels,
        probabilities[:, 1]
    )

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "specificity":
            float(specificity),

        "f1":
            float(f1),

        "auroc":
            float(auroc),

        "pr_auc":
            float(pr_auc),

        "confusion_matrix": {

            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    labels,
    predictions,
    probabilities,
    path
):

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label",
            "prediction",
            "probability_normal",
            "probability_covid"
        ])

        for i in range(
            len(labels)
        ):

            writer.writerow([

                int(labels[i]),

                int(predictions[i]),

                float(
                    probabilities[i][0]
                ),

                float(
                    probabilities[i][1]
                )
            ])


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation(
    model_config,
    scenario_config,
    experiment_id
):

    # ========================================================
    # CONFIG
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

    model_name = (
        model_config_data[
            "model"
        ]["name"]
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # PATHS
    # ========================================================

    experiment_dir = (

        Path(
            base_config[
                "output"
            ]["root"]
        )

        / experiment_id

        / model_name
    )

    training_dir = (
        experiment_dir
        / "training"
    )

    evaluation_dir = (
        experiment_dir
        / "evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model_path = (
        training_dir
        / "best_model.pth"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model(
        model_config_data
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    model = model.to(device)

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
    # TEST
    # ========================================================

    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Device: {device}"
    )

    labels, predictions, probabilities = (
        predict(
            model,
            test_loader,
            device
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\nTest Results")
    print("-" * 40)

    print(
        f"Accuracy    : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Specificity : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1          : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"AUROC       : "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"PR-AUC      : "
        f"{metrics['pr_auc']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = metrics[
        "confusion_matrix"
    ]

    print("\nConfusion Matrix")
    print("-" * 40)

    print(
        f"TN: {cm['TN']}"
    )

    print(
        f"FP: {cm['FP']}"
    )

    print(
        f"FN: {cm['FN']}"
    )

    print(
        f"TP: {cm['TP']}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    class_names = [
        "Normal",
        "COVID"
    ]

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    save_json(
        report,
        evaluation_dir
        / "classification_report.json"
    )

    # ========================================================
    # ROC
    # ========================================================

    fpr, tpr, roc_thresholds = (
        roc_curve(
            labels,
            probabilities[:, 1]
        )
    )

    roc_data = {

        "fpr":
            fpr.tolist(),

        "tpr":
            tpr.tolist(),

        "thresholds":
            roc_thresholds.tolist(),

        "auroc":
            float(
                auc(fpr, tpr)
            )
    }

    save_json(
        roc_data,
        evaluation_dir
        / "roc_curve.json"
    )

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(
            labels,
            probabilities[:, 1]
        )
    )

    pr_data = {

        "precision":
            precision_curve.tolist(),

        "recall":
            recall_curve.tolist(),

        "thresholds":
            pr_thresholds.tolist(),

        "pr_auc":
            float(
                auc(
                    recall_curve,
                    precision_curve
                )
            )
    }

    save_json(
        pr_data,
        evaluation_dir
        / "pr_curve.json"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    save_json(
        metrics,
        evaluation_dir
        / "metrics.json"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    save_predictions(
        labels,
        predictions,
        probabilities,
        evaluation_dir
        / "predictions.csv"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    evaluation_summary = {

        "experiment_id":
            experiment_id,

        "model":
            model_name,

        "test_samples":
            int(len(labels)),

        "metrics":
            metrics,

        "files": {

            "metrics":
                "metrics.json",

            "classification_report":
                "classification_report.json",

            "roc_curve":
                "roc_curve.json",

            "pr_curve":
                "pr_curve.json",

            "predictions":
                "predictions.csv"
        }
    }

    save_json(
        evaluation_summary,
        evaluation_dir
        / "evaluation_summary.json"
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Output: {evaluation_dir}"
    )

    print("=" * 70)

    return evaluation_summary
