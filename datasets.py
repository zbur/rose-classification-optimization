"""PyTorch datasets and loaders for training / validation."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD
from preprocess import items_for, load_rgb, preprocess_pil


def train_transform(size):
    """Moderate augmentation (slide 3 notes: control overfitting)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RoseDataset(Dataset):
    def __init__(self, items, size, train):
        self.items = items
        self.size = size
        self.tf = train_transform(size) if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        img = load_rgb(item["abs_path"])
        if self.tf is not None:
            x = self.tf(img)
        else:
            # Same preprocessing as deployment (preprocess.py).
            x = torch.from_numpy(preprocess_pil(img, self.size))
        return x, item["label"]


def make_loader(split, subset, size, batch_size, train, workers=4):
    ds = RoseDataset(items_for(split, subset), size, train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)
