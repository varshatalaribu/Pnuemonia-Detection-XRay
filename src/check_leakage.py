"""Check the train / validation / test splits for data leakage.

Run once after downloading the dataset:
    python -m src.check_leakage

It checks, using the exact same split that training uses:
  1. Filenames whose patient ID can't be read (those images might belong to a
     child who also has images in another split, and we couldn't tell).
  2. Patients who appear in more than one split.
  3. Identical image files in more than one split.
  4. Near-identical images (the same X-ray re-saved or slightly re-encoded)
     in more than one split.
  5. The same image carrying two different labels.

Exits with code 1 if anything that counts as leakage is found.
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import Config
from src.dataset import (CLASS_NAMES, PATIENT_ID_PATTERNS, collect_samples, patient_id,
                         split_by_patient, strict_patient_id)

MAX_EXAMPLES = 10


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def difference_hash(path: Path, size: int = 16) -> str:
    """A 256-bit fingerprint that survives resizing and re-compression.

    Shrink to a tiny grayscale image and record whether each pixel is brighter
    than its right-hand neighbour. Two copies of the same X-ray give the same
    fingerprint; two different X-rays almost never do at this size.
    """
    with Image.open(path) as img:
        small = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def has_known_pattern(path: Path) -> bool:
    return any(p.match(path.name) for p in PATIENT_ID_PATTERNS)


def show(examples, indent="    "):
    for line in examples[:MAX_EXAMPLES]:
        print(f"{indent}{line}")
    if len(examples) > MAX_EXAMPLES:
        print(f"{indent}... and {len(examples) - MAX_EXAMPLES} more")


def find_duplicates(splits, hash_fn, description):
    """Return (cross_split_groups, conflicting_label_groups)."""
    by_hash = defaultdict(list)
    total = sum(len(s) for s in splits.values())
    done = 0
    for split_name, samples in splits.items():
        for path, label in samples:
            by_hash[hash_fn(path)].append((split_name, path, label))
            done += 1
            if done % 1000 == 0:
                print(f"    hashed {done}/{total} images ({description})")

    cross_split, label_conflicts = [], []
    for group in by_hash.values():
        if len(group) < 2:
            continue
        if len({split for split, _, _ in group}) > 1:
            cross_split.append(group)
        if len({label for _, _, label in group}) > 1:
            label_conflicts.append(group)
    return cross_split, label_conflicts


def format_group(group):
    return "  <->  ".join(f"{split}:{path.parent.name}/{path.name}" for split, path, _ in group)


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Check the dataset splits for data leakage")
    parser.add_argument("--data_dir", type=Path, default=cfg.data_dir)
    parser.add_argument("--val_fraction", type=float, default=cfg.val_fraction)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--skip_near_duplicates", action="store_true",
                        help="skip check 4 (it opens every image, ~1-2 minutes)")
    args = parser.parse_args()

    pooled = collect_samples(args.data_dir / "train") + collect_samples(args.data_dir / "val")
    train, val = split_by_patient(pooled, args.val_fraction, args.seed)
    test = collect_samples(args.data_dir / "test")
    splits = {"train": train, "val": val, "test": test}
    problems = []

    print("Splits (same as training uses):")
    for name, samples in splits.items():
        n_patients = len({patient_id(p) for p, _ in samples})
        print(f"  {name:>5}: {len(samples):5d} images, {n_patients:4d} patients")

    # 1. Unreadable patient IDs
    print("\n[1] Filenames without a recognisable patient ID")
    unmatched = [f"{name}: {p.parent.name}/{p.name}"
                 for name, samples in splits.items() for p, _ in samples if not has_known_pattern(p)]
    if unmatched:
        print(f"  WARNING: {len(unmatched)} images. Each is treated as its own patient, "
              "so leakage through them can't be ruled out by ID:")
        show(unmatched)
        problems.append("unrecognised filenames")
    else:
        print("  OK: every filename has a patient ID")

    # 2. Patient overlap
    print("\n[2] Patients in more than one split")
    ids = {name: {strict_patient_id(p) for p, _ in samples} for name, samples in splits.items()}
    loose_ids = {name: {patient_id(p) for p, _ in samples} for name, samples in splits.items()}
    strict_to_loose = {strict_patient_id(p): patient_id(p)
                       for samples in splits.values() for p, _ in samples}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(ids[a] & ids[b])
            # Same "personN" number on both sides, but not the same patient under the strict ID.
            number_only = len((loose_ids[a] & loose_ids[b]) - {strict_to_loose[x] for x in shared})
            if shared:
                print(f"  LEAK: {len(shared)} patients in both {a} and {b}:")
                show(shared)
                problems.append(f"patient overlap {a}/{b}")
            else:
                print(f"  OK: {a} / {b} share no patients")
            if number_only > 0:
                print(f"  NOTE: {number_only} person numbers appear in both {a} and {b}, but only as "
                      "bacteria in one and virus in the other (very likely different children)")

    # 3 + 4. Identical and near-identical images
    checks = [(file_hash, "identical files")]
    if not args.skip_near_duplicates:
        checks.append((difference_hash, "near-identical images"))
    for number, (hash_fn, description) in enumerate(checks, start=3):
        print(f"\n[{number}] {description.capitalize()} in more than one split")
        cross, conflicts = find_duplicates(splits, hash_fn, description)
        if cross:
            print(f"  LEAK: {len(cross)} groups of {description} span splits:")
            show([format_group(g) for g in cross])
            problems.append(f"{description} across splits")
        else:
            print(f"  OK: no {description} across splits")
        if conflicts:
            print(f"  WARNING: {len(conflicts)} groups of {description} have different labels:")
            show([format_group(g) for g in conflicts])
            problems.append(f"{description} with conflicting labels")

    print("\n" + "=" * 60)
    if problems:
        print("RESULT: problems found -> " + "; ".join(problems))
        print("Fix these (or explain them in the README) before reporting test results.")
        raise SystemExit(1)
    print("RESULT: no leakage found between train, val and test.")
    print("Not checkable from the data: a child with both a NORMAL and a PNEUMONIA X-ray")
    print(f"({CLASS_NAMES[0]} and {CLASS_NAMES[1]} folders number patients separately).")


if __name__ == "__main__":
    main()
