import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

DATA_ROOT = os.path.join('stanford_dogs_dataset', 'images', 'Images')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class SplitConfig:
    train_per_class: int = 40
    val_per_class: int = 5
    test_per_class: int = 5

SPLIT = SplitConfig()


def list_breeds() -> List[str]:
    breeds = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    breeds.sort()
    return breeds


def collect_images(breed: str) -> List[str]:
    folder = os.path.join(DATA_ROOT, breed)
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    return images


def build_split_dataframe(limit_per_class: int = 50) -> pd.DataFrame:
    rows = []
    for breed in list_breeds():
        imgs = collect_images(breed)[:limit_per_class]
        random.Random(SEED).shuffle(imgs)
        train = imgs[:SPLIT.train_per_class]
        val = imgs[SPLIT.train_per_class:SPLIT.train_per_class + SPLIT.val_per_class]
        test = imgs[SPLIT.train_per_class + SPLIT.val_per_class:SPLIT.train_per_class + SPLIT.val_per_class + SPLIT.test_per_class]
        for p in train:
            rows.append({'path': p, 'label': breed, 'split': 'train'})
        for p in val:
            rows.append({'path': p, 'label': breed, 'split': 'val'})
        for p in test:
            rows.append({'path': p, 'label': breed, 'split': 'test'})
    return pd.DataFrame(rows)


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_to_idx[row.label]
        return img, label


def build_model(num_classes: int):
    # Lightweight backbone (MobileNetV3-Small) for speed; replaceable.
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def accuracy(output, target):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        return (preds == target).float().mean().item()


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def train(model, train_loader, val_loader, epochs=5, lr=3e-4, wd=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val = 0.0
    best_path = 'cnn_best.pt'

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        running_loss = 0.0
        running_acc = 0.0
        batches = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            acc = accuracy(out, y)
            running_loss += loss.item()
            running_acc += acc
            batches += 1
            pbar.set_postfix({'loss': running_loss / batches, 'acc': running_acc / batches})
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion)
        train_metrics = {'loss': running_loss / batches, 'accuracy': running_acc / batches}
        record = {'epoch': epoch, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['accuracy'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy']}
        history.append(record)
        print(f"Val acc: {val_metrics['accuracy']:.4f}")
        if val_metrics['accuracy'] > best_val:
            best_val = val_metrics['accuracy']
            torch.save(model.state_dict(), best_path)
            print('Saved new best model.')
    return history, best_path


def main():
    df = build_split_dataframe(limit_per_class=50)
    breeds = sorted(df.label.unique())
    label_to_idx = {b: i for i, b in enumerate(breeds)}

    train_df = df[df.split == 'train']
    val_df = df[df.split == 'val']
    test_df = df[df.split == 'test']

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_ds = ImageDataset(train_df, label_to_idx, train_tfms)
    val_ds = ImageDataset(val_df, label_to_idx, eval_tfms)
    test_ds = ImageDataset(test_df, label_to_idx, eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(num_classes=len(breeds)).to(device)

    print(f'Training on {len(train_ds)} images across {len(breeds)} classes using device {device}.')
    history, best_path = train(model, train_loader, val_loader, epochs=8)

    # Load best and evaluate test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print('Test metrics:', test_metrics)

    pd.DataFrame(history).to_csv('cnn_training_history.csv', index=False)
    with open('cnn_test_metrics.txt', 'w') as f:
        f.write(str(test_metrics))

    # Per-class accuracy
    model.eval()
    class_correct = {c: 0 for c in breeds}
    class_total = {c: 0 for c in breeds}
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            for yi, pi in zip(y.cpu().tolist(), preds.cpu().tolist()):
                class_total[breeds[yi]] += 1
                if yi == pi:
                    class_correct[breeds[yi]] += 1
    per_class = []
    for c in breeds:
        per_class.append({'breed': c, 'accuracy': class_correct[c] / class_total[c] if class_total[c] else 0.0})
    pd.DataFrame(per_class).to_csv('cnn_per_class_accuracy.csv', index=False)
    print('Saved per-class accuracy to cnn_per_class_accuracy.csv')

if __name__ == '__main__':
    main()
