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



    phoneme = ["b","d","f","g","dʒ","k","l","m","n","p","r","s","t","v","w","z","ʒ","tʃ","ʃ","θ","ð","ŋ","j","æ","eɪ",
               "e","i:","ɪ","aɪ","ɒ","oʊ","ʊ","ʌ","u:","ɔɪ","aʊ","ə","eə","ɑ:","ɜ:","ɔ:","ɪə","ʊə"]

    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    f = open("hjhj.txt", "r")
    content = f.read()
    content_list = content.split(",")
    f.close()
    punc = []
    index = 0
    for row in content:
        if index % 2 == 0:
          punc.append(row)
        index += 1
    print(punc)

    def word_length_vis (self):
        dict = {}
        for index in range(len(self.data)):
            p, g = self.data[index]
            if len(g) in dict:
                dict[len(g)] += 1
            else:
                dict[len(g)] = 1
        Y = []
        for ele in dict.keys():
            Y.appned(dict[ele])
        X = dict.keys()
        fig = plt.figure()
        plt.bar(X, Y, 0.4, color="green")
        plt.xlabel("* length")
        plt.ylabel("numbers of * length")
        plt.title("bar chart")
        plt.savefig("word_length.jpg")


    def ave_letter_vis (self):
        dict = {}
        for index in range(len(self.data)):
            p, g = self.data[index]
            lowString = g.lower()
            for char in lowString:
                if char in dict:
                    dict[char] += 1
                else:
                    dict[char] = 1
        Y = []
        for ele in dict.keys():
            Y.appned(dict[ele]/len(self.data))
        X = dict.keys()
        fig = plt.figure()
        plt.bar(X, Y, 0.4, color="green")
        plt.xlabel("letter")
        plt.ylabel("average letter in each word")
        plt.title("bar chart")
        plt.savefig("ave_letter")