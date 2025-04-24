# Stand‑alone script for training a binary CNN classifier (melanoma vs. benign)
# on a dermatoscopic image dataset arranged like:
#
# Example run:
# py -3.11 train.py --data_dir data_dir --epochs 5 --batch_size 64 --lr 3e-4 --output model_best.pth


import argparse
import os
from pathlib import Path
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN for melanoma vs. benign classification."
    )
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--output", default="best_model.pth", type=str)
    return parser.parse_args()


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=test_tfm)
    test_ds = datasets.ImageFolder(Path(data_dir) / "test", transform=test_tfm)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, labels) * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)

    # Final test evaluation
    model.load_state_dict(torch.load(args.output, map_location=device))
    test_loss, test_acc = evaluate(model, test_dl, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Best model saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
