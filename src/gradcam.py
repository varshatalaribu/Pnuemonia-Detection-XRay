"""Grad-CAM heatmaps: where in the image is the model looking? (Shortcut check 2)

Grad-CAM takes the last layer of feature maps (a coarse grid, 7 x 7 for
ResNet-18) and asks, for each map, how much raising it would raise the
PNEUMONIA score. Maps that push toward pneumonia are weighted up, the rest
down, and the weighted sum becomes a heatmap laid over the image.

Warm colours on the lungs = the model uses the lungs. Warm colours on the
edges, corners, text markers or outside the body = a shortcut.

The picture shows the image exactly as the model saw it (after squashing or
cropping), so you can also see how squashed each X-ray was.

Usage:
    python -m src.gradcam --model_name resnet18                   # 12 most confident false alarms
    python -m src.gradcam --model_name resnet18 --show true_positives
    python -m src.gradcam --run_name resnet18_crop --show false_positives
    python -m src.gradcam --model_name resnet18 --images path/a.jpeg path/b.jpeg
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, collect_samples, get_transforms
from src.shape_check import load_run
from src.utils import get_device


def features_and_logits(model, model_name: str, x):
    """Run the network by hand so we can keep the last feature maps."""
    if model_name == "resnet18":
        m = model
        f = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        fmap = m.layer4(m.layer3(m.layer2(m.layer1(f))))
        logits = m.fc(torch.flatten(m.avgpool(fmap), 1))
    elif model_name == "densenet121":
        fmap = F.relu(model.features(x))
        logits = model.classifier(torch.flatten(F.adaptive_avg_pool2d(fmap, 1), 1))
    elif model_name == "simple_cnn":
        fmap = model.features(x)
        logits = model.classifier(model.pool(fmap))
    else:
        raise ValueError(f"Grad-CAM not set up for {model_name}")
    return fmap, logits


def grad_cam(model, model_name, x):
    """Return (heatmap in 0..1 at input size, pneumonia probability)."""
    with torch.enable_grad():
        fmap, logits = features_and_logits(model, model_name, x)
        score = logits[0, 1] - logits[0, 0]  # evidence for pneumonia over normal
        grads, = torch.autograd.grad(score, fmap)
    weights = grads.mean(dim=(2, 3), keepdim=True)          # importance of each map
    cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))  # keep pro-pneumonia evidence
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max() if cam.max() > 0 else cam
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    return cam.detach().cpu().numpy(), prob


def to_display(x):
    """Undo the normalisation so the X-ray looks normal again."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x[0].cpu() * std + mean).clamp(0, 1).mean(0).numpy()


def pick_test_images(model, cfg, device, show: str, count: int):
    """Choose test images of one kind, most confident first."""
    transform = get_transforms(cfg.image_size, False, cfg.resize_mode)
    samples = collect_samples(cfg.data_dir / "test")
    want_label = {"false_positives": 0, "true_negatives": 0,
                  "true_positives": 1, "false_negatives": 1}[show]
    want_flagged = show in ("false_positives", "true_positives")
    scored = []
    with torch.no_grad():
        for path, label in samples:
            if label != want_label:
                continue
            x = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            prob = torch.softmax(model(x), dim=1)[0, 1].item()
            if (prob >= 0.5) == want_flagged:
                scored.append((prob, path))
    # most confident mistakes / most confident correct answers first
    scored.sort(reverse=want_flagged)
    return [p for _, p in scored[:count]]


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for a trained model")
    parser.add_argument("--model_name", default="resnet18")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--images", nargs="*", type=Path, help="specific images (optional)")
    parser.add_argument("--show", default="false_positives",
                        choices=["false_positives", "true_negatives", "true_positives", "false_negatives"])
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    run_dir = args.output_dir / (args.run_name or args.model_name)
    device = get_device()
    model, cfg = load_run(run_dir, device)

    if args.images:
        paths, name = args.images, "gradcam_selected"
    else:
        paths, name = pick_test_images(model, cfg, device, args.show, args.count), f"gradcam_{args.show}"
    if not paths:
        raise SystemExit(f"No test images of type '{args.show}' at threshold 0.5.")

    transform = get_transforms(cfg.image_size, False, cfg.resize_mode)
    cols = 4
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, paths):
        image = Image.open(path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        cam, prob = grad_cam(model, cfg.model_name, x)
        ax.imshow(to_display(x), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        w, h = image.size
        ax.set_title(f"{path.parent.name[:4]} {path.name[:18]}\nP(pneumonia) {prob:.2f} · w/h {w / h:.2f}",
                     fontsize=8)

    fig.suptitle(f"Grad-CAM · {run_dir.name} · resize: {cfg.resize_mode} · {name.replace('gradcam_', '')}",
                 fontsize=11)
    fig.tight_layout()
    out = run_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out} ({len(paths)} images)")


if __name__ == "__main__":
    main()
