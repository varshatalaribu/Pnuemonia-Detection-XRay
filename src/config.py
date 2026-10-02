from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data/chest_xray")
    output_dir: Path = Path("outputs")

    model_name: str = "resnet18"

    image_size: int = 224
    # How images become square: "squash" stretches them (original behaviour);
    # "crop" keeps proportions and cuts the centre square, so aspect ratio
    # can't be used as a clue. See README, "Shortcut experiment".
    resize_mode: str = "squash"
    # Chance of a slight random blur on each training image (0 = never).
    blur_prob: float = 0.0
    batch_size: int = 32
    val_fraction: float = 0.15
    num_workers: int = 2

    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    seed: int = 42

    # Folder name under outputs/. Empty = use model_name. Set it to keep
    # experiments apart, e.g. --run_name resnet18_crop.
    run_name: str = ""

    def run_dir(self) -> Path:
        return self.output_dir / (self.run_name or self.model_name)

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
