import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

from src.config import Config
from src.dataset import CLASS_NAMES, make_dataloaders
from src.model import build_model
from src.utils import get_device


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity_recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = np.argmax(tpr - fpr)
    return float(min(thresholds[best], 1.0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument("--model_name", default="resnet18")
    parser.add_argument("--run_name", default="",
                        help="folder under outputs/ if training used --run_name")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    label = args.run_name or args.model_name
    run_dir = args.output_dir / label
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)

    saved = checkpoint["config"]
    cfg = Config(**{k: Path(v) if k.endswith("_dir") else v for k, v in saved.items()})

    device = get_device()
    model = build_model(cfg.model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    print(f"Loaded {label} ({cfg.model_name}, resize: {cfg.resize_mode}) from epoch {checkpoint['epoch']}")

    _, val_loader, test_loader, _ = make_dataloaders(cfg)
    val_true, val_prob = predict_probabilities(model, val_loader, device)
    test_true, test_prob = predict_probabilities(model, test_loader, device)

    tuned = youden_threshold(val_true, val_prob)
    results = {
        "model": label,
        "test_at_0.5": compute_metrics(test_true, test_prob, 0.5),
        "test_at_tuned_threshold": compute_metrics(test_true, test_prob, tuned),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(results, indent=2, default=float))

    for name in ("test_at_0.5", "test_at_tuned_threshold"):
        m = results[name]
        print(f"\n== {name} (threshold = {m['threshold']:.3f}) ==")
        for key in ("accuracy", "sensitivity_recall", "specificity", "precision", "f1", "auc_roc"):
            print(f"  {key:<20} {m[key]:.3f}")
        print(f"  confusion matrix     {m['confusion_matrix']}")

    y_pred = (test_prob >= tuned).astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix(test_true, y_pred, labels=[0, 1]),
                                  display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{label} — test set (threshold {tuned:.2f})")
    plt.savefig(run_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(test_true, test_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {results['test_at_0.5']['auc_roc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random guess")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title(f"ROC curve — {label}")
    plt.legend()
    plt.savefig(run_dir / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nSaved test_metrics.json, confusion_matrix.png and roc_curve.png to {run_dir}/")


if __name__ == "__main__":
    main()
