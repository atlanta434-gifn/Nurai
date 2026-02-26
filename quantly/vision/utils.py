from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import random
import shutil
import csv

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def list_class_folders(images_root: Path) -> List[Path]:
    if not images_root.exists():
        return []
    return sorted([p for p in images_root.iterdir() if p.is_dir() and not p.name.startswith(".")])

def list_images_in_folder(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])

def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def write_csv(rows: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def train_val_split(items: List[Path], val_ratio: float, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 2 else 0
    val = items[:n_val]
    train = items[n_val:]
    return train, val
