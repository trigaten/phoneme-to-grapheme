# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Phoneme to Grapheme Conversion with a Recurrent Generative Model 
# This project will discuss...

# %%
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# find phoneme vocabulary
data = pd.read_csv("phonemes-words.csv")
phonemes_col = data["phonemes"]
graphemes_col = data["graphemes"]
phonemes = ['0', '1']
graphemes = ['0', '1']

for word in phonemes_col:
    # print(word)
    for phoneme in word:
        if phoneme not in phonemes:
            phonemes.append(phoneme)
for word in graphemes_col:
    # print(word)
    for grapheme in word:
        if grapheme not in graphemes:
            graphemes.append(grapheme)
print(phonemes)


# %%

# one hot encodes the word: returns an array of one hot encoded characters
def nemes_to_1_hot_seq(string, nemes="phonemes"):
    string = '0' + string + '1'
    l = phonemes if nemes == "phonemes" else graphemes
    seq = []
    for i in string:
        vec = [0] * len(l)
        vec[l.index(i)] = 1
        seq.append(vec)

    return torch.FloatTensor([seq])

def one_hot_to_nemes(arr, nemes="phonemes"):
    seq = []
    l = phonemes if nemes == "phonemes" else graphemes
    for hot in arr:
        x = torch.argmax(hot)
        seq.append(l[x])
    return seq

class P2GDataset(Dataset):
    def __init__(self, phoneme_file, device):
        df = pd.read_csv(phoneme_file)
        self.data = df.drop(df[df["phonemes"].map(len) > 7].index)

        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p, g = self.data.iloc[idx]
        return nemes_to_1_hot_seq(p, nemes = "phonemes").to(self.device), nemes_to_1_hot_seq(g, nemes = "graphemes").long()


# %%
layer_size = 512
# define model architecture
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(len(phonemes), layer_size, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(2 * layer_size, layer_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        # push vector through encoder
        out, hidden = self.encoder(x)
        # hidden is [4, 1, layer_size]
        # this is because of bidirectionality * double stacked
        # we want to grab the "highest" layers from the forwards and backwards directions
        # dim 1 because hidden[3] and hidden[4] are both [1, layer_size] and we 
        # want a single batch that has 2*layer_size values
        # print(hidden.shape)
        hc = torch.cat((hidden[2], hidden[3]), dim=1)
        hidden_for_init = self.fc(hc)

        # return context vector
        return out, hidden_for_init

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 from encoder state and 1 from decoder state (since not bidirectional)
        self.energy = nn.Sequential(
            nn.Linear(3*layer_size, layer_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(layer_size, 1, bias=False)
    
    def forward(self, encoder_hiddens, decoder_hidden):
        # encoder_hiddens is [1, L, layer_size*2] bc bidirectional
        # decoder_hidden is [1, layer_size]
        # 1 bc using batch first 
        num_encoder_hiddens = encoder_hiddens.shape[1]

        # make it [1,1,layer_size]
        decoder_hidden = torch.unsqueeze(decoder_hidden, 0)

        # repeat along second dim to get [4, 1, layer_size]
       
        decoder_hiddens = decoder_hidden.squeeze(0).repeat(1, num_encoder_hiddens, 1)
        
        inputs = torch.cat((encoder_hiddens, decoder_hiddens), 2)

        energy = self.energy(inputs)

        attention = self.attention(energy)

        # want a distribution of attention that sums to 1
        return F.softmax(attention, dim=2)
        
class Decoder(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.decoder = nn.GRU(len(graphemes) + 2*layer_size, layer_size, batch_first=True, bidirectional=False, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(layer_size*3 + len(graphemes), len(graphemes))
        )
        
    def forward(self, input, hidden_layer, encoder_hiddens):
        """
        Since this function gets called once at a time rather than taking in
        a sequence of vectors, we need to pass it the last output. This will be just
        a vector of numbers that can be converted to the embedding representing that last output
        """
        # [1,1,layer_size]
        attention_vals = self.attention(encoder_hiddens, hidden_layer)
        # encoder_hiddens [1,L,layer_size]
        # this just multiplies each attention value against the appropriate vector
        # and sums the weighted vectors
        # will be [1, 1, layer_size]
        
        attention_vals = attention_vals.permute(0, 2, 1)
        attended = torch.bmm(attention_vals, encoder_hiddens)
#         torch.Size([1, 6, 1])
# torch.Size([1, 6, 1024])
        input = torch.cat((attended, input), dim=2)
        out, hidden = self.decoder(input, hidden_layer)
        # out[1] to get top hidden layer
        input_for_fc = torch.cat((input, out), dim = 2)
        # print("H")
        # (1x1564 and 1075x28)
        return self.fc(input_for_fc), hidden

class seq2seq(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder = Encoder()
        self.attention = Attention()
        self.decoder = Decoder(self.attention)
        self.device = device
    
    def forward(self, in_seq, out_seq, tf_ratio=0.5):
        out_len = out_seq.shape[1]
        # storing the outputs of the sequence
        outputs = torch.zeros(out_len, 1, len(graphemes)).to(self.device)

        out_for_at, hidden = self.encoder(in_seq)
        hidden = hidden.unsqueeze(0)
        out_seq = out_seq.squeeze(0)

        input = out_seq[0].unsqueeze(0).unsqueeze(0).float().to(device)
        
        for i in range(1, out_len):
            out, hidden = self.decoder(input, hidden, out_for_at)
            outputs[i] = out

            if random.random() > tf_ratio:
                # teacher forcing (make next input what the current output token should be)
                input = out_seq[i].unsqueeze(0).unsqueeze(0).float().to(device)
            else:
                x = input.argmax(1)[0]
                input = torch.zeros(1, 1, len(graphemes)).to(self.device)
                input[0][0][x] = 1
                
        return outputs

    def pred_new(self, in_seq):
        encoder_out_for_at, hidden = self.encoder(in_seq)
        hidden = hidden.unsqueeze(0)
        input = torch.zeros(1, 1, len(graphemes)).to(self.device)
        outs = []
        while True:
            out, hidden = self.decoder(input, hidden, encoder_out_for_at)
            outs.append(out)
            if len(outs) > 30:
                break
            x = input.argmax(1)[0]
            input = torch.zeros(1, 1, len(graphemes)).to(self.device)
            input[0][0][x] = 1
            if one_hot_to_nemes(out) == ['1']:
                break
        return outs


# %%
"""training"""
from torch.utils.data import random_split
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
model = seq2seq(device).to(device)
# what a beautiful architecture
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()
dataset = P2GDataset("phonemes-words.csv", device)
train, test = random_split(dataset, [35000, len(dataset)-35000])
dataloader = DataLoader(dataset=train, batch_size=1)
print(len(test))

def get_0_1_accuracy(test_set, model):
    correct = 0
    dataloader = DataLoader(dataset=test_set, batch_size=1)
    for (in_seq, out_seq) in dataloader:
        # print(out_seq.shape)
        # break
        prediction = model.pred_new(in_seq[0])
        # print(in_seq[0].shape)
        true = "".join(one_hot_to_nemes(out_seq[0][0], "graphemes"))[1:-1]
        print(true)
        # print(one_hot_to_nemes(prediction, "graphemes"))
        # print(prediction)
        pred = "".join(one_hot_to_nemes(prediction, "graphemes"))[0:-1]
        print(pred)
        
        if true == pred:
            correct+= 1
    if correct == 0:
        return correct
    return correct/len(test_set)


# %%
print(sum(p.numel() for p in model.parameters()))


# %%
avg_losses = []
writer = SummaryWriter("tensorboard_data")

_, mini_test = random_split(test, [20, len(test)-20])
# 15 quite good
for epoch in range(30):
    tot_loss = 0
    for (in_seq, out_seq) in dataloader:
        in_seq = in_seq.squeeze(0)
        out_seq = out_seq.squeeze(0)
        model_output = model(in_seq, out_seq)
        model_output = model_output[1:]
        model_output = model_output.squeeze(1)
        out_seq = out_seq.squeeze(0)[1:]
        loss = loss_func(model_output, out_seq.argmax(1).to(device))
        tot_loss+=loss.detach().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    tot_loss/=len(dataset)
    writer.add_scalar("tensorboard_data/acc", get_0_1_accuracy(mini_test, model), epoch)
    writer.add_scalar("tensorboard_data", loss.detach().item(), epoch)
    avg_losses.append(tot_loss)


# %%
dataset = P2GDataset("data.csv", "cuda")
p, g = dataset[0]

print(one_hot_to_nemes(p[0], "phonemes"))
p.shape


# %%
model.eval()


def print_preds(path):
    global p
    print(one_hot_to_nemes(p[0], "phonemes"))
    s = model.pred_new(p)
    
    print(one_hot_to_nemes(s, "graphemes"))

# print_preds("data.csv")
print(get_0_1_accuracy(test, model))
# 36 great for train set
# print(test[0])
# print(one_hot_to_graphemes(torch.FloatTensor([[3,2,1],[0,0,1],[0,0,1]])))


# %%
model.encoder.encoder.weight_ih_l0


