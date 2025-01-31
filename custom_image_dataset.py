import torch
from torch.utils.data import Dataset
import os
import pandas as pd


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return {
            'img': image,  # Image tensor
            'label': label,  # Ground truth labels
            # 'img_metas': {}
        }