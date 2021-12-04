import torch

phonemes = [
    'ō', 
    'ē'
]

def phonemes_to_1_hot_seq(string):
    seq = []
    for i in string:
        vec = [0] *  len(phonemes)
        vec[phonemes.index(i)] = 1
        seq.append(vec)

    return torch.FloatTensor([seq])

print(phonemes_to_1_hot_seq("ōō"))