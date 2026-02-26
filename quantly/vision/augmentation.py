from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from PIL import Image, ImageEnhance

@dataclass
class AugmentConfig:
    flips: bool = True
    rotate: bool = True
    brightness: bool = True
    contrast: bool = True
    max_rotate_deg: int = 25
    brightness_range: Tuple[float, float] = (0.75, 1.25)
    contrast_range: Tuple[float, float] = (0.75, 1.25)
    seed: int = 42

def augment_one(img: Image.Image, rng: random.Random, cfg: AugmentConfig) -> Image.Image:
    out = img.copy()
    if cfg.flips and rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if cfg.rotate and rng.random() < 0.7:
        deg = rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        out = out.rotate(deg, resample=Image.BILINEAR, expand=False)
    if cfg.brightness and rng.random() < 0.7:
        f = rng.uniform(*cfg.brightness_range)
        out = ImageEnhance.Brightness(out).enhance(f)
    if cfg.contrast and rng.random() < 0.7:
        f = rng.uniform(*cfg.contrast_range)
        out = ImageEnhance.Contrast(out).enhance(f)
    return out

def augment_folder(class_folder: Path, out_folder: Path, copies_per_image: int, cfg: AugmentConfig) -> List[Path]:
    rng = random.Random(cfg.seed)
    out_folder.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    for img_path in sorted(class_folder.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or not img_path.is_file():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        stem = img_path.stem
        # keep original
        out_orig = out_folder / f"{stem}__orig{img_path.suffix.lower()}"
        img.save(out_orig)
        created.append(out_orig)
        for k in range(copies_per_image):
            aug = augment_one(img, rng, cfg)
            out_p = out_folder / f"{stem}__aug{k+1:02d}.png"
            aug.save(out_p)
            created.append(out_p)
    return created
