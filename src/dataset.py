import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def collect_samples(split_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected folder {class_dir} — did you download and unzip the dataset? See README.md"
            )
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


# Many children have several X-rays. The patient ID is in the filename:
#   PNEUMONIA: person1_bacteria_1.jpeg, person1_virus_6.jpeg   -> "person1"
#   NORMAL:    IM-0115-0001.jpeg, NORMAL2-IM-0666-0001.jpeg      -> "IM-0115", "NORMAL2-IM-0666"
PATIENT_ID_PATTERNS = [
    re.compile(r"^(person\d+)_", re.IGNORECASE),
    re.compile(r"^((?:NORMAL2-)?IM-\d+)-", re.IGNORECASE),
]


def patient_id(path: Path) -> str:
    """Return an ID shared by all X-rays of the same child.

    The class folder is part of the ID because the NORMAL and PNEUMONIA folders
    number their patients separately. If a filename doesn't match the known
    patterns, the image is treated as its own patient.
    """
    for pattern in PATIENT_ID_PATTERNS:
        match = pattern.match(path.name)
        if match:
            return f"{path.parent.name}/{match.group(1)}"
    return f"{path.parent.name}/{path.stem}"


INFECTION_TYPE = re.compile(r"^person\d+_(bacteria|virus)_", re.IGNORECASE)


def strict_patient_id(path: Path) -> str:
    """Patient ID that also includes the infection type for PNEUMONIA images.

    In this dataset the "personN" numbers appear to be counted separately for
    bacterial and viral cases: person1_bacteria_* and person1_virus_* are very
    likely two different children (see README, "Design decisions").
    The train/val split still groups them together to be safe; this stricter
    ID is only used to check for overlap with the official test set.
    """
    match = INFECTION_TYPE.match(path.name)
    base = patient_id(path)
    return f"{base}/{match.group(1).lower()}" if match else base


def split_by_patient(samples, val_fraction: float, seed: int):
    """Split into train/val so that no child appears in both.

    Splitting by image instead would let the model be validated on children it
    has already trained on, which makes validation scores look better than they
    are (and those scores choose the saved epoch and the decision threshold).
    StratifiedGroupKFold keeps the NORMAL/PNEUMONIA ratio similar in both parts.
    """
    labels = [label for _, label in samples]
    groups = [patient_id(path) for path, _ in samples]
    n_splits = max(2, round(1 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(samples, labels, groups))
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


class ChestXRayDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def resize_steps(image_size: int, resize_mode: str) -> list:
    """Make every image image_size x image_size.

    squash: stretch to a square. Wide images get squeezed sideways, so the
            original aspect ratio stays visible to the model. In this dataset
            pneumonia images are wider than normal ones, so that is a clue
            unrelated to the lungs.
    crop:   shrink so the short side fits, then cut out the centre square.
            Proportions are kept and every image ends up the same shape, so
            aspect ratio can't be used. Wide images lose some of their sides.
    """
    if resize_mode == "squash":
        return [transforms.Resize((image_size, image_size))]
    if resize_mode == "crop":
        return [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    raise ValueError(f"Unknown resize_mode '{resize_mode}'. Choose squash or crop.")


def get_transforms(image_size: int, train: bool, resize_mode: str = "squash",
                   blur_prob: float = 0.0):
    steps = resize_steps(image_size, resize_mode)
    if train:
        steps += [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        if blur_prob > 0:
            # Normal images are ~2.5x larger than pneumonia images before resizing,
            # which can leave differences in sharpness. Random blur weakens that clue.
            steps.append(transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=blur_prob))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def compute_class_weights(samples: list[tuple[Path, int]]) -> torch.Tensor:
    counts = Counter(label for _, label in samples)
    total = len(samples)
    weights = [total / (len(CLASS_NAMES) * counts[c]) for c in range(len(CLASS_NAMES))]
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(cfg: Config):
    pooled = collect_samples(cfg.data_dir / "train") + collect_samples(cfg.data_dir / "val")
    train_samples, val_samples = split_by_patient(pooled, cfg.val_fraction, cfg.seed)
    test_samples = collect_samples(cfg.data_dir / "test")

    train_tf = get_transforms(cfg.image_size, True, cfg.resize_mode, cfg.blur_prob)
    eval_tf = get_transforms(cfg.image_size, False, cfg.resize_mode)
    train_ds = ChestXRayDataset(train_samples, train_tf)
    val_ds = ChestXRayDataset(val_samples, eval_tf)
    test_ds = ChestXRayDataset(test_samples, eval_tf)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    class_weights = compute_class_weights(train_samples)
    return train_loader, val_loader, test_loader, class_weights


def describe(samples, name):
    counts = Counter(CLASS_NAMES[label] for _, label in samples)
    patients = len({patient_id(path) for path, _ in samples})
    print(f"{name:>6}: {len(samples):5d} images from {patients:4d} patients  ->  {dict(counts)}")


def check_patient_overlap(splits: dict) -> None:
    """Print how many patients appear in more than one split (should be 0).

    Uses the strict ID, so person1_bacteria and person1_virus count as different
    children. Run `python -m src.check_leakage` for the full check.
    """
    ids = {name: {strict_patient_id(path) for path, _ in samples} for name, samples in splits.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = ids[a] & ids[b]
            status = "OK" if not shared else "WARNING"
            print(f"patients in both {a} and {b}: {len(shared)}  [{status}]")


if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader, test_loader, class_weights = make_dataloaders(cfg)
    describe(train_loader.dataset.samples, "train")
    describe(val_loader.dataset.samples, "val")
    describe(test_loader.dataset.samples, "test")
    check_patient_overlap({
        "train": train_loader.dataset.samples,
        "val": val_loader.dataset.samples,
        "test": test_loader.dataset.samples,
    })
    print("class weights [NORMAL, PNEUMONIA]:", class_weights.tolist())

    images, labels = next(iter(train_loader))
    print("one batch -> images:", tuple(images.shape), " labels:", tuple(labels.shape))
