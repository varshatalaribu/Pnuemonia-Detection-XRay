"""Predict pneumonia for one or more new chest X-ray images.

Usage:
    python -m src.predict path/to/xray.jpeg
    python -m src.predict img1.jpeg img2.jpeg --model_name densenet121

Educational project only. Not for clinical use.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.dataset import CLASS_NAMES, get_transforms
from src.model import build_model
from src.utils import get_device


def load_model(run_dir: Path, device):
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = build_model(cfg["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, cfg


def load_threshold(run_dir: Path) -> tuple[float, str]:
    """Use the threshold tuned on validation data by evaluate.py, if it exists."""
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        return float(metrics["test_at_tuned_threshold"]["threshold"]), "tuned on validation"
    return 0.5, "default (run src.evaluate to tune it)"


@torch.no_grad()
def predict(model, image_path: Path, cfg: dict, device) -> float:
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(cfg["image_size"], False, cfg.get("resize_mode", "squash"))
    x = transform(image).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0, 1].item()


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia from chest X-ray images")
    parser.add_argument("images", nargs="+", type=Path, help="one or more image files")
    parser.add_argument("--model_name", default="resnet18")
    parser.add_argument("--run_name", default="",
                        help="folder under outputs/ if training used --run_name")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    run_dir = args.output_dir / (args.run_name or args.model_name)
    if not (run_dir / "best_model.pt").exists():
        raise SystemExit(f"No trained model at {run_dir}/best_model.pt. "
                         f"Train one first: python -m src.train --model_name {args.model_name}")

    device = get_device()
    model, cfg = load_model(run_dir, device)
    threshold, source = load_threshold(run_dir)
    print(f"Model: {run_dir.name} | decision threshold {threshold:.3f} ({source})\n")

    for path in args.images:
        prob = predict(model, path, cfg, device)
        label = CLASS_NAMES[int(prob >= threshold)]
        print(f"{path.name}: {prob:.1%} chance of pneumonia  ->  {label}")

    print("\nEducational project only. Not a diagnosis.")


if __name__ == "__main__":
    main()
