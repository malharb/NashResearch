
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
  def __init__(self, num_encoder_layers = 5, num_heads = 5, d_model = 32, num_metrics = 5, dropout = 0.05, batch_size = 64, token_length = 20160):
    super(Transformer, self).__init__()

    self.num_encoder_layers = num_encoder_layers
    self.num_heads = num_heads
    self.d_model = d_model
    self.num_metrics = num_metrics
    self.dropout = dropout
    self.batch_size = batch_size
    self.token_length = token_length

    # input embedding layer
    self.input_embedding_layer = nn.Sequential(
        nn.Linear(in_features = num_metrics, out_features = d_model * 2),
        nn.ReLU(),
        nn.Linear(in_features = d_model * 2, out_features = d_model),
        nn.ReLU(),
    )

    self.positional_encoding = PositionalEncoding(d_model = 64)

  def forward(self, x):
    # x is of the form (batch size, num of tokens, num features)
    self.input_embedding = self.input_embedding_layer(x)






class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len = 10000):
    super(PositionalEncoding, self).__init__()

    pe = torch.zeros(max_len, d_model)

    position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) # get position indices
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)) # get positional coefficients

    pe[:,0::2] = torch.sin(position * div_term) # set even indices of embeddings
    pe[:,1::2] = torch.cos(position * div_term) # set odd indices of embeddings
    pe = pe.unsqueeze(0) # broadcast to account for batch size

    self.register_buffer('pe', pe)

  def forward(self, x):
    x += self.pe[:,:x.size(1), :] # adds positional encodings to input embeddings
    return x
