import cv2
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class BipedalDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        images = sorted([img for img in os.listdir(data_path) if img.endswith(".png")], key=lambda x: int(x[4:-4]))
        images = np.array([cv2.imread(os.path.join(data_path, img_path)) for img_path in images])
        images = np.transpose(images, (0, 3, 1, 2)).astype('float32')
        images /= 255.
        if images.shape[1] == 135:
            print(f"failed to load {self.data_paths[idx]}")
        states = np.load(os.path.join(data_path, "states.npy"), allow_pickle=True).astype('float32')
        actions = np.load(os.path.join(data_path, "actions.npy"), allow_pickle=True).astype('float32')
        return torch.as_tensor(images), torch.as_tensor(states), torch.as_tensor(actions)

