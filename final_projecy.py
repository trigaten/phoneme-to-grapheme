import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd


class Phoneme_Dataset(Dataset):
    def __init__(self, phoneme_file, transform=None, target_transform=None):
        self.data = pd.read_csv("phoneme.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p, g = self.data[idx]
        return self.nemes_to_1_hot_seq(p, nemes = "phonemes"), self.nemes_to_1_hot_seq(g, nemes = "phonemes")

    

