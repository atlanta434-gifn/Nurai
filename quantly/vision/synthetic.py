from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import random
from PIL import Image, ImageDraw

def _rand_color(rng: random.Random):
    return (rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225))

def generate_shapes_dataset(out_root: Path, n_per_class: int = 40, image_size: int = 128, seed: int = 1) -> List[Tuple[str, Path]]:
    """
    Creates a tiny demo dataset:
      - circle
      - square
      - triangle
    into:
      out_root/<class>/*.png
    Returns list of (label, path).
    """
    rng = random.Random(seed)
    classes = ["circle", "square", "triangle"]
    out_root.mkdir(parents=True, exist_ok=True)
    created = []
    for cls in classes:
        (out_root/cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.new("RGB", (image_size, image_size), (15, 15, 15))
            d = ImageDraw.Draw(img)
            pad = rng.randint(10, 25)
            x0, y0 = pad, pad
            x1, y1 = image_size - pad, image_size - pad
            fill = _rand_color(rng)
            if cls == "circle":
                d.ellipse([x0, y0, x1, y1], fill=fill)
            elif cls == "square":
                d.rectangle([x0, y0, x1, y1], fill=fill)
            else:
                # triangle
                p1 = (image_size//2, y0)
                p2 = (x0, y1)
                p3 = (x1, y1)
                d.polygon([p1, p2, p3], fill=fill)
            # add light noise dots
            for _ in range(rng.randint(20, 60)):
                x = rng.randint(0, image_size-1)
                y = rng.randint(0, image_size-1)
                img.putpixel((x,y), _rand_color(rng))
            fp = out_root/cls/f"{cls}_{i:04d}.png"
            img.save(fp)
            created.append((cls, fp))
    return created
