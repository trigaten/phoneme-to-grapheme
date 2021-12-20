# %% [markdown]
# # Phoneme to Grapheme Conversion with a Recurrent Generative Model 
# We hope the reader will appreciate the attempts at humor built in for subjective reasons.
# This project will discuss...

# %%
# necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import random
import pandas as pd
# for logging
from torch.utils.tensorboard import SummaryWriter

# compute phoneme/grapheme vocabularies
data = pd.read_csv("phonemes-words.csv")
phonemes_col = data["phonemes"]
graphemes_col = data["graphemes"]
# vocabularies contain 0 and 1 as start and end tokens
phonemes = ['0', '1']
graphemes = ['0', '1']

for word in phonemes_col:
    for phoneme in word:
        if phoneme not in phonemes:
            phonemes.append(phoneme)
for word in graphemes_col:
    for grapheme in word:
        if grapheme not in graphemes:
            graphemes.append(grapheme)
            
# wow, there are lot of different phonemes!
print(phonemes)

# %%

def nemes_to_1_hot_seq(string, nemes="phonemes"):
    """one hot encodes the word according to either
    the phoneme or grapheme list
        ::returns:: pytorch tensor of one hot encoded characters
    """
    string = '0' + string + '1'
    l = phonemes if nemes == "phonemes" else graphemes
    seq = []
    for i in string:
        vec = [0] * len(l)
        vec[l.index(i)] = 1
        seq.append(vec)

    return torch.FloatTensor([seq])

def one_hot_to_nemes(arr, nemes="phonemes"):
    """converts a 1-hot encoding back to characters"""
    seq = []
    l = phonemes if nemes == "phonemes" else graphemes
    for hot in arr:
        x = torch.argmax(hot)
        seq.append(l[x])
    return seq


class P2GDataset(Dataset):
    """Pytorch dataset object for sampling the dataset"""
    def __init__(self, phoneme_file, device):
        df = pd.read_csv(phoneme_file)
        self.data = df
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p, g = self.data.iloc[idx]
        # 1-hot encoding
        return nemes_to_1_hot_seq(p, nemes = "phonemes").to(self.device), nemes_to_1_hot_seq(g, nemes = "graphemes").long()


# %% [markdown]
# # Architecture
# The model changed signifigantly over time. We tried the following combinations:
# 
# Single layer GRU encoder+decoder
# 
# Double/Triple stacked GRU encoder+decoder
# 
# Single layer LSTM encoder+decoder
# 
# We also varied hidden sizes, testing 512 or 1024
# 
# We settled on the following architecture (hidden size 512):
# 
# Double bidirectional stacked GRU encoder -> linear layer which accepts the last forward/backward hidden layers and converts them to a vector the size of a single hidden layer -> unstacked unidirectional GRU decoder with Bahdanau attention.
# 

# %%
# hidden layer size
layer_size = 512
# define model architecture
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(len(phonemes), layer_size, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Sequential(
            # takes final forwards and backwards hidden states
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
        # map energy vectors to single values
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
        # decoder GRU takes in previous output word, attention vector, current hidden state
        self.decoder = nn.GRU(len(graphemes) + 2*layer_size, layer_size, batch_first=True)
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
        attention_vals = attention_vals.permute(0, 2, 1)

        # encoder_hiddens [1,L,layer_size]
        # this just multiplies each attention value against the appropriate vector
        # and sums the weighted vectors
        # will be [1, 1, layer_size]
        attended = torch.bmm(attention_vals, encoder_hiddens)
        input = torch.cat((attended, input), dim=2)
        out, hidden = self.decoder(input, hidden_layer)
        # out[1] to get top hidden layer
        input_for_fc = torch.cat((input, out), dim = 2)

        return self.fc(input_for_fc), hidden

class seq2seq(nn.Module):
    """The seq2seq model itself"""
    def __init__(self, device):
        super().__init__()
        # instantiate encoder and decoder with attention
        self.encoder = Encoder()
        self.decoder = Decoder(Attention())
        self.device = device
    
    def forward(self, in_seq, out_seq, tf_ratio=0.5):
        """
        :param tf_ratio: is the teacher forcing ratio. It decides how frequently
        the model receives its own previously predicted token as opposed to the
        known correct token.
        """
        out_len = out_seq.shape[1]
        # storing the outputs of the sequence
        outputs = torch.zeros(out_len, 1, len(graphemes)).to(self.device)

        out_for_at, hidden = self.encoder(in_seq)
        hidden = hidden.unsqueeze(0)
        out_seq = out_seq.squeeze(0)

        # perform an embarassing amount of data conversions
        input = out_seq[0].unsqueeze(0).unsqueeze(0).float().to(device)
        
        # for each token in known out sequence (except the first)
        for i in range(1, out_len):
            out, hidden = self.decoder(input, hidden, out_for_at)
            outputs[i] = out

            if random.random() > tf_ratio:
                # teacher forcing (make next input what the current output token should be)
                input = out_seq[i].unsqueeze(0).unsqueeze(0).float().to(device)
            else:
                # use previously output token
                x = input.argmax(1)[0]
                input = torch.zeros(1, 1, len(graphemes)).to(self.device)
                input[0][0][x] = 1
                
        return outputs

    def pred_new(self, in_seq):
        """Method to predict the output sequence for a previously unseen
        input sequence. The main difference between this function and forward
        is that this function only stops decoding when the model produces and
        end token
        """
        encoder_out_for_at, hidden = self.encoder(in_seq)
        hidden = hidden.unsqueeze(0)
        input = torch.zeros(1, 1, len(graphemes)).to(self.device)
        outs = []
        while True:
            out, hidden = self.decoder(input, hidden, encoder_out_for_at)
            outs.append(out)
            # in case not hitting end token
            if len(outs) > 50:
                break
            x = input.argmax(1)[0]
            input = torch.zeros(1, 1, len(graphemes)).to(self.device)
            input[0][0][x] = 1
            if one_hot_to_nemes(out) == ['1']:
                break
        return outs

# %%
# initialize optimizer/loss func/hyperparams

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
model = seq2seq(device).to(device)
# what a beautiful architecture
print("Model architecture ", model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()
dataset = P2GDataset("phonemes-words.csv", device)
# train on 70000 words
train, test = random_split(dataset, [70000, len(dataset)-70000])
dataloader = DataLoader(dataset=train, batch_size=1)
print("train size ", len(train))
print("test size ", len(test))

def get_0_1_accuracy(test_set, model):
    """method to compute 1-WER accuracy AKA what % of test_set does model get
    exactly correct."""
    correct = 0
    dataloader = DataLoader(dataset=test_set, batch_size=1)
    for (in_seq, out_seq) in dataloader:
        prediction = model.pred_new(in_seq[0])
        true = "".join(one_hot_to_nemes(out_seq[0][0], "graphemes"))[1:-1]
        print(true)
        pred = "".join(one_hot_to_nemes(prediction, "graphemes"))[0:-1]
        print(pred)
        
        if true == pred:
            correct+= 1
    if correct == 0:
        return correct
    return correct/len(test_set)


# %%
print("# of model parameters: ", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# Wow 10 million params! This model has more trainable parameters than tonnes of potatoes France produced in 2016! (absolutely no semantic relation). This might take a while to train, so make sure to use a NVIDIA GeForce RTX 2080 Ti :).

# %%
writer = SummaryWriter("tensorboard_data")
# get a mini testing batch to check model accuracy on the test set
# throughout training
# NOTE: this is not a validation set
_, mini_test = random_split(test, [20, len(test)-20])

# begin training loop
for epoch in range(EPOCHS):
    tot_loss = 0
    for (in_seq, out_seq) in dataloader:
        # batch size of 1
        in_seq = in_seq.squeeze(0)
        out_seq = out_seq.squeeze(0)
        # perform inference
        model_output = model(in_seq, out_seq)
        # dont compute loss using first token of in/out sequence
        model_output = model_output[1:]
        model_output = model_output.squeeze(1)
        out_seq = out_seq.squeeze(0)[1:]
        # compute loss
        loss = loss_func(model_output, out_seq.argmax(1).to(device))
        # record loss
        tot_loss+=loss.detach().item()
        # accumulate gradients
        loss.backward()
        # step and clear grads
        optimizer.step()
        optimizer.zero_grad()
    
    tot_loss/=len(train)
    # record current accuracy on test set and average loss
    writer.add_scalar("tensorboard_data/acc", get_0_1_accuracy(mini_test, model), epoch)
    writer.add_scalar("tensorboard_data/loss", tot_loss, epoch)

# %%
# turn dropout off
model.eval()

with torch.no_grad():
    print("Test accuracy: " + str(get_0_1_accuracy(test, model)))

torch.save(model, "THEMODEL")
# %% [markdown]
# Well that accuracy is... okay. It might be better than the average human (when faced with 10s of thousands of words), but thats still a lot of error. 

# %% [markdown]
# ![alt text](secret_ingredient.jpg)

# %%
model.encoder.encoder.weight_ih_l0

# %% [markdown]
# # Resources
# If the reader would like more resources related to this topic:
# 
# For learning the basics of RNNs, LSTMs, GRUs, attention (including Bahdanau), and seq2seq architectures, these resources are good:
# 
# https://www.deeplearningbook.org/contents/rnn.html
# 
# https://d2l.ai/chapter_recurrent-modern/seq2seq.html
# 
# https://d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html
# 
# For more comprehensive tutorials that walk through the full deep learning process (including varied seq2seq architectures such as transformer), this is a good resource:
# 
# https://github.com/bentrevett/pytorch-seq2seq
# 
# These papers discuss grapheme->phoneme conversion with deep learning. This is an easier problem, but still requires complex models for high success rates:
# 
# https://arxiv.org/abs/2004.06338
# 
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43264.pdf


