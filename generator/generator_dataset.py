import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle

class GeneratorDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, text, style = self.data[idx]
        return img, text, style
