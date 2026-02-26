from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from .utils import list_class_folders, list_images_in_folder, ensure_empty_dir, train_val_split, write_csv

def build_augmented_dataset(images_root: Path, output_root: Path, copies_per_image: int,
                            aug_cfg, val_ratio: float = 0.2, seed: int = 42) -> Dict[str, Path]:
    """
    Creates:
      output_root/augmented/<class>/*.png
      output_root/splits/train/<class>/*.png
      output_root/splits/val/<class>/*.png
      output_root/splits/train.csv and val.csv

    Returns paths dictionary.
    """
    from .augmentation import augment_folder

    ensure_empty_dir(output_root)
    augmented_root = output_root / "augmented"
    train_root = output_root / "splits" / "train"
    val_root = output_root / "splits" / "val"
    augmented_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    rows_train: List[Dict] = []
    rows_val: List[Dict] = []

    for class_dir in list_class_folders(images_root):
        cls = class_dir.name
        out_cls_aug = augmented_root / cls
        created = augment_folder(class_dir, out_cls_aug, copies_per_image=copies_per_image, cfg=aug_cfg)

        # split augmented images
        train_imgs, val_imgs = train_val_split(created, val_ratio=val_ratio, seed=seed)
        (train_root/cls).mkdir(parents=True, exist_ok=True)
        (val_root/cls).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            dst = train_root/cls/p.name
            shutil.copy2(p, dst)
            rows_train.append({"image_path": str(dst.as_posix()), "label": cls})
        for p in val_imgs:
            dst = val_root/cls/p.name
            shutil.copy2(p, dst)
            rows_val.append({"image_path": str(dst.as_posix()), "label": cls})

    write_csv(rows_train, output_root/"splits"/"train.csv")
    write_csv(rows_val, output_root/"splits"/"val.csv")

    return {
        "augmented_root": augmented_root,
        "train_root": train_root,
        "val_root": val_root,
        "train_csv": output_root/"splits"/"train.csv",
        "val_csv": output_root/"splits"/"val.csv",
        "output_root": output_root
    }
