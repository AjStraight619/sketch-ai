import os
import torch
from torch.utils.data import Dataset
import numpy as np

class SketchDataset(Dataset):
    def __init__(self, data_dir, class_names, idxs=None, transform=None):
        self.data = []
        self.labels = []
        for idx, class_name in enumerate(class_names):
            imgs = np.load(os.path.join(data_dir, f"{class_name}.npy"))
            if idxs is not None:
                imgs = imgs[idxs[idx]]
            self.data.append(imgs)
            self.labels.extend([idx] * imgs.shape[0])
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels)
        self.transform = transform  # Pass a torchvision transform or None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.uint8)
        # img shape: (28, 28); ToPILImage expects shape (H, W) or (H, W, C)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img[None, ...] / 255.0, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
