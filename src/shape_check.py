"""Does the model use image shape instead of the lungs? (Shortcut check 1)

In this dataset, NORMAL X-rays are larger and closer to square, while PNEUMONIA
X-rays are smaller and wider. With resize_mode "squash", a wide image gets
squeezed sideways, so shape survives resizing and can act as a clue.

This script runs a trained model on the test set and splits the results by the
ORIGINAL aspect ratio (width / height) of each image:
  - For NORMAL images: how often each group is wrongly flagged as pneumonia.
  - For PNEUMONIA images: how often each group is caught.
If wide normal images are flagged much more often than square ones, the model
is probably using shape.

It also trains a tiny model on image size alone (no pixels) as a reference:
how much could a model get just from shape?

Usage:
    python -m src.shape_check --model_name resnet18
    python -m src.shape_check --run_name resnet18_crop
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.dataset import collect_samples, make_dataloaders
from src.evaluate import predict_probabilities
from src.model import build_model
from src.utils import get_device

# Groups by original width / height. Training NORMAL images have a median of
# about 1.22, PNEUMONIA about 1.49.
ASPECT_BINS = [(0.0, 1.3, "square-ish (< 1.30)"),
               (1.3, 1.4, "medium (1.30-1.40)"),
               (1.4, 9.9, "wide (>= 1.40)")]


def image_sizes(samples):
    sizes = []
    for path, _ in samples:
        with Image.open(path) as img:  # reads only the header, fast
            sizes.append(img.size)
    return np.array(sizes, dtype=float)


def load_run(run_dir: Path, device):
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    saved = checkpoint["config"]
    cfg = Config(**{k: Path(v) if k.endswith("_dir") else v for k, v in saved.items()})
    model = build_model(cfg.model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), cfg


def main():
    parser = argparse.ArgumentParser(description="Check whether a model relies on image shape")
    parser.add_argument("--model_name", default="resnet18")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = args.output_dir / (args.run_name or args.model_name)
    device = get_device()
    model, cfg = load_run(run_dir, device)
    print(f"Run: {run_dir.name} ({cfg.model_name}, resize: {cfg.resize_mode}) | threshold {args.threshold}")

    _, _, test_loader, _ = make_dataloaders(cfg)
    y_true, y_prob = predict_probabilities(model, test_loader, device)
    sizes = image_sizes(test_loader.dataset.samples)
    aspect = sizes[:, 0] / sizes[:, 1]
    flagged = y_prob >= args.threshold

    report = {"run": run_dir.name, "resize_mode": cfg.resize_mode,
              "threshold": args.threshold, "groups": []}
    print(f"\n{'original shape':<22}{'NORMAL':>8}{'false alarms':>16}{'PNEUMONIA':>12}{'caught':>12}")
    for lo, hi, name in ASPECT_BINS:
        in_bin = (aspect >= lo) & (aspect < hi)
        normal, sick = in_bin & (y_true == 0), in_bin & (y_true == 1)
        fa = flagged[normal].mean() if normal.any() else float("nan")
        caught = flagged[sick].mean() if sick.any() else float("nan")
        print(f"{name:<22}{normal.sum():>8}{fa:>15.1%} {sick.sum():>11}{caught:>11.1%}")
        report["groups"].append({"group": name, "normal_n": int(normal.sum()),
                                 "false_alarm_rate": float(fa), "pneumonia_n": int(sick.sum()),
                                 "sensitivity": float(caught)})

    normal = y_true == 0
    corr = float(np.corrcoef(aspect[normal], y_prob[normal])[0, 1])
    report["normal_aspect_vs_probability_correlation"] = corr
    print(f"\nNORMAL images: correlation between width/height and predicted pneumonia "
          f"probability = {corr:+.2f}")
    print("  (near 0 = shape doesn't drive the score; clearly positive = wider looks sicker)")

    # Reference: a model that sees only the image size, never the pixels.
    train_samples = collect_samples(cfg.data_dir / "train") + collect_samples(cfg.data_dir / "val")
    tr_sizes = image_sizes(train_samples)
    features = lambda s: np.column_stack([s[:, 0], s[:, 1], s[:, 0] / s[:, 1]])
    size_model = GradientBoostingClassifier(random_state=0).fit(
        features(tr_sizes), [label for _, label in train_samples])
    size_auc = roc_auc_score(y_true, size_model.predict_proba(features(sizes))[:, 1])
    model_auc = roc_auc_score(y_true, y_prob)
    report.update(size_only_auc=float(size_auc), model_auc=float(model_auc))
    print(f"\nTest AUC from image size alone (no pixels): {size_auc:.3f}")
    print(f"Test AUC of this model:                     {model_auc:.3f}")

    (run_dir / "shape_check.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {run_dir}/shape_check.json")


if __name__ == "__main__":
    main()
