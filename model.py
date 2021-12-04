import torch
import torch.nn as nn
import utils

class seq2seq(nn.Module):
    def __init__(self, device, ):
        super().__init__()
        self.encoder = Encoder(self.post_embeddings)
        self.decoder = Decoder(self.comment_embeddings)
        self.device = device
    
    def forward(self, in_seq, out_seq, tf_ratio=0.5):
        out, context = self.encoder(in_seq)
        
        return context

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(30, 1200, 2, batch_first=True, bidirectional=False)
        
    def forward(self, x):
        # push vector through encoder
        out, h_n = self.encoder(x)

        # return context vector
        return h_n

class Decoder(nn.Module):
    def __init__(self, comment_embeddings):
        super().__init__()
        self.comment_embeddings = comment_embeddings
        self.decoder = nn.GRU(30, 1200, 2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(1200, comment_embeddings.num_embeddings)

    def forward(self, hidden_layer, last_output_word):
        """
        Since this function gets called once at a time rather than taking in
        a sequence of vectors, we need to pass it the last output. This will be just
        a vector of numbers that can be converted to the embedding representing that last output
        """
        out, h_n = self.decoder(context, embedded)

        return self.fc(h_n), h_n
