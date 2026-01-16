# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CarDamageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Safe image loading
        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = (
            row[["broken_glass", "dent", "scratch", "wreck"]]
            .astype(float)
            .values
        )

        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels
