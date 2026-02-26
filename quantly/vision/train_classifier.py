from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@dataclass
class TrainConfig:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-3
    device: str = "cpu"

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def train_classifier(train_dir: Path, val_dir: Path, out_model_path: Path, cfg: TrainConfig) -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=len(train_ds.classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        acc = evaluate(model, val_loader, str(device))
        best_acc = max(best_acc, acc)

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": train_ds.classes,
        "image_size": cfg.image_size,
    }, out_model_path)

    return {"val_accuracy": float(best_acc), "num_classes": float(len(train_ds.classes)), "train_samples": float(len(train_ds)), "val_samples": float(len(val_ds))}
