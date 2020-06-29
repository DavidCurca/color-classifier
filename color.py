import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class ColorDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 0:3])
        self.y = torch.from_numpy(xy[:, 3:4])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
