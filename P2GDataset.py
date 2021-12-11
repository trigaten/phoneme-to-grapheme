from torch.utils.data import Dataset
import pandas as pd

class P2GDataset(Dataset):
    def __init__(self, phoneme_file):
        self.data = pd.read_csv(phoneme_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p, g = self.data[idx]
        return self.nemes_to_1_hot_seq(p, nemes = "phonemes"), self.nemes_to_1_hot_seq(g, nemes = "phonemes")

    

