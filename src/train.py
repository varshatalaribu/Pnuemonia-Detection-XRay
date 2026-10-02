import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.config import Config
from src.dataset import make_dataloaders
from src.model import build_model, count_parameters
from src.utils import get_device, set_seed


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train a pneumonia classifier")
    for field, default in cfg.__dict__.items():
        arg_type = Path if isinstance(default, Path) else type(default)
        parser.add_argument(f"--{field}", type=arg_type, default=default)
    args = parser.parse_args()
    return Config(**vars(args))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


def plot_history(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(title="Loss (lower is better)", xlabel="epoch")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set(title="Accuracy", xlabel="epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print(f"Device: {device}")
    print(f"Run: {run_dir} | resize: {cfg.resize_mode} | blur_prob: {cfg.blur_prob}")
    train_loader, val_loader, _, class_weights = make_dataloaders(cfg)
    print(f"Train images: {len(train_loader.dataset)} | Val images: {len(val_loader.dataset)}")
    print(f"Class weights [NORMAL, PNEUMONIA]: {[round(w, 2) for w in class_weights.tolist()]}")

    model = build_model(cfg.model_name).to(device)
    print(f"Model: {cfg.model_name} ({count_parameters(model):,} trainable parameters)")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch},
                       run_dir / "best_model.pt")
            marker = "  <- best so far, saved"

        print(f"Epoch {epoch:2d}/{cfg.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"{time.time() - start:.0f}s{marker}")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, run_dir / "training_curves.png")
    print(f"\nDone. Best validation loss: {best_val_loss:.4f}. Files saved in {run_dir}/")
    run_arg = f"--run_name {cfg.run_name}" if cfg.run_name else f"--model_name {cfg.model_name}"
    print(f"Next step: python -m src.evaluate {run_arg}")


if __name__ == "__main__":
    main()
