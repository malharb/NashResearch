
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
    self.input_layer_norm = nn.LayerNorm(self.d_model)
    self.positional_encoding_layer = PositionalEncoding(d_model = 64)

  def forward(self, x):
    # x is of the form (batch size, num of tokens, num features)
    self.input_embedding = self.input_embedding_layer(x)
    self.input_embedding = self.input_layer_norm(self.input_embedding)
    self.input_embedding = self.positional_encoding_layer(self.input_embedding)

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

class EncoderBlock(nn.Module):

  def __init__(self, num_tokens, d_model):
    super(EncoderBlock, self).__init__()

    self.num_tokens = num_tokens
    self.d_model = d_model
    
    num_heads = 4
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.norm1 = nn.LayerNorm(d_model)

    self.ffn = nn.Sequential(
        nn.Linear(in_features = d_model, out_features = 4 * d_model),
        nn.ReLU(),
        nn.Linear(in_features = 4 * d_model, out_features = d_model)
    ) 
    self.norm2 = nn.LayerNorm(d_model)

  def forward(self, x):

    # x is input embeddings of shape (batch_size, num_tokens, d_model)

    mha_out = self.mha(x)
    residual_addition_attention = x + mha_out
    norm_attention = self.norm1(residual_addition_attention)
    
    ffn_output = self.ffn(norm_attention)
    residual_addition_ffn = norm_attention + ffn_output
    
    encoder_output = self.norm2(residual_addition_ffn)

    return encoder_output


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads # get dimension for each head

    self.query_heads = nn.ModuleList([nn.Linear(self.d_model, self.head_dim) for head in range(num_heads)])
    self.key_heads = nn.ModuleList([nn.Linear(self.d_model, self.head_dim) for head in range(num_heads)])
    self.value_heads = nn.ModuleList([nn.Linear(self.d_model, self.head_dim) for head in range(num_heads)])

    self.w_matrix = nn.Linear(self.d_model, self.d_model)

  def forward(self, input_matrix):

    batch_size = input_matrix.size(0)
    num_tokens = input_matrix.size(1)

    query_matrices = [query_matrix_weights(input_matrix) for query_matrix_weights in self.query_heads]
    key_matrices = [key_matrix_weights(input_matrix) for key_matrix_weights in self.key_heads]
    value_matrices = [value_matrix_weights(input_matrix) for value_matrix_weights in self.value_heads]

    z_matrices = []

    for query_matrix, key_matrix, value_matrix in zip(query_matrices, key_matrices, value_matrices):

       score_matrix = torch.matmul(query_matrix, key_matrix.transpose(-2,-1)) / math.sqrt(self.head_dim)
       attention_weights = torch.softmax(score_matrix, dim = -1)
       attention_matrix = torch.matmul(attention_weights, value_matrix)

       z_matrices.append(attention_matrix)

    multihead_z_concat = torch.cat(z_matrices, dim = -1)
    encoder_attention_output = self.w_matrix(multihead_z_concat)

    return encoder_attention_output

class ConvolutionLayer(nn.Module):
  def __init__(self, d_model, vertical, num_features):
    super(ConvolutionLayer, self).__init__()

    self.num_features = num_features
    self.kernel_size = (vertical, d_model)
    self.stride = (vertical, 1)
  
    self.conv_layer = nn.Conv2d(in_channels = 1, out_channels = self.num_features, kernel_size = self.kernel_size, stride = self.stride)

  def forward(self, x):

    batch_size = x.size(0)
    num_tokens = x.size(1)

    print(f"before 4d transform, x shape: {x.shape}")
    x = x.unsqueeze(0).permute(1,0,2,3)
    print(f"after 4d transform, x shape: {x.shape}")

    convolution_output = self.conv_layer(x)
    print(f"convolution output shape: {convolution_output.shape}")
    n_condensed_tokens = convolution_output.size(2)

    condensed_encoding = (convolution_output.permute(0,2,1,3)).view(batch_size, n_condensed_tokens, self.num_features)
    print(f"condensed_encoding expected shape: {(batch_size, num_tokens, self.num_features)}, actual shape : {condensed_encoding.shape}")

    return condensed_encoding
