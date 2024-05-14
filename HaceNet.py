
import torch
import torch.nn as nn
import math

class HaceNet(nn.Module):
  def __init__(self, num_encoder_levels = 4, num_heads = 5, d_model_in_list = [32,32,(32,48),32], d_model_out_list = [10,20,(30,32),40], token_in_list = [1440, 288, (144, 216), 84], token_out_list = [144, 72, (24, 36), 84], num_metrics = 5, dropout = 0.05, batch_size = 64, token_length = 20160):
    super(HaceNet, self).__init__()

    self.num_encoder_levels = num_encoder_levels
    self.num_heads = num_heads
    self.d_model_in_list = d_model_in_list 
    self.d_model_out_list = d_model_out_list
    self.num_metrics = num_metrics
    self.dropout = dropout
    self.batch_size = batch_size
    self.token_in_list = token_in_list  
    self.token_out_list = token_out_list 

    # input embedding layer
    self.input_encodings = nn.ModuleList([InputEncoding(d_model_in = self.num_metrics, d_model_out = self.d_model_in_list[0])])
    for dmo_index in range(len(self.d_model_out_list) - 1):
      if (dmo_index == 1):
        for sub_index in range(2):
          self.input_encodings.append(InputEncoding(d_model_in = self.d_model_out_list[dmo_index], d_model_out = self.d_model_in_list[dmo_index + 1][sub_index]))
        continue

      if (dmo_index == 2):
        for sub_index in range(2):
          self.input_encodings.append(InputEncoding(d_model_in = self.d_model_out_list[dmo_index][sub_index], d_model_out = self.d_model_in_list[dmo_index + 1]))
        continue
      self.input_encodings.append(InputEncoding(d_model_in = self.d_model_out_list[dmo_index], d_model_out = self.d_model_in_list[dmo_index + 1]))

    # position encoding layer
    self.positional_encodings = nn.ModuleList()
    for k in range(len(token_in_list)):
      if (k == 2):
        for j in range(2):
          self.positional_encodings.append(PositionalEncoding(d_model = self.d_model_in_list[k][j], max_len = self.token_in_list[k][j]))
        continue

      self.positional_encodings.append(PositionalEncoding(d_model = self.d_model_in_list[k], max_len = token_in_list[k]))

    #  segment encodings
    self.segment_encodings_lvl_1 = []
    for segment in range(2):
      self.segment_encodings_lvl_1.append(nn.Parameter(torch.randn(1, self.d_model_in_list[1])))

    self.segment_encodings_lvl_2_s1 = []
    self.segment_encodings_lvl_2_s2 = []

    for segment in range(2):
      self.segment_encodings_lvl_2_s1.append(nn.Parameter(torch.randn(1, self.d_model_in_list[2][0])))
    for segment in range(3):
      self.segment_encodings_lvl_2_s2.append(nn.Parameter(torch.randn(1, self.d_model_in_list[2][1])))

    self.segment_encodings_lvl_3 = []
    for segment in range(3):
      self.segment_encodings_lvl_3.append(nn.Parameter(torch.randn(1, self.d_model_in_list[3])))

    #hace layers
    self.hace_layer_level = [0,1,2,2,3]
    self.hace_blocks = nn.ModuleList()

    for i in range(self.num_encoder_levels):
      if (i == 2):
        for j in range(2):
          self.hace_blocks.append(HACEBlock(num_tokens_in = token_in_list[i][j],
                                  num_tokens_out = token_out_list[i][j],
                                  d_model_in = d_model_in_list[i][j],
                                  d_model_out = d_model_out_list[i][j]))
        continue
      self.hace_blocks.append(HACEBlock(num_tokens_in = token_in_list[i],
                                      num_tokens_out = token_out_list[i],
                                      d_model_in = d_model_in_list[i],
                                      d_model_out = d_model_out_list[i]))

    # post processing
    self.postprocessing_linear_1 = nn.Linear(in_features = d_model_out_list[(self.num_encoder_levels) - 1], out_features = 42)
    self.postprocessing_linear_2 = nn.Linear(in_features = 42, out_features = 16)
    self.postprocessing_linear_3 = nn.Linear(in_features = 16, out_features = 1)

    self.pre_output_linear = nn.Linear(in_features = self.token_out_list[-1], out_features = 1)
    self.output_sigmoid = nn.Sigmoid()

  def forward(self, X):

 #  print("Processing level 1")
    L1 = X
    for day_matrix_idx in range(14):
      L1[day_matrix_idx] = (self.input_encodings[0])(L1[day_matrix_idx])
      L1[day_matrix_idx] = (self.positional_encodings[0])(L1[day_matrix_idx])
      L1[day_matrix_idx] = (self.hace_blocks[0])(L1[day_matrix_idx])

  #  print("Processing level 2")
    L2 = []
    L2_idx = 0
    segment_l1_length = 144
    for day_matrix_idx in range(0,14,2):
      L2.append(torch.cat((L1[day_matrix_idx], L1[day_matrix_idx + 1]), dim = 1))
      L2[L2_idx] = (self.input_encodings[1])(L2[L2_idx])
      (L2[L2_idx])[:, :segment_l1_length, :] += self.segment_encodings_lvl_1[0]
      (L2[L2_idx])[:, segment_l1_length:, :] += self.segment_encodings_lvl_1[1]
      L2[L2_idx] = (self.positional_encodings[1])(L2[L2_idx])
      L2[L2_idx] = (self.hace_blocks[1])(L2[L2_idx])
      L2_idx += 1

   # print("Processing level 3")

    L3 = []
    segment_l3_s1_length = 72
    segment_l3_s2_length = 72

    L3.append(torch.cat((L2[0], L2[1]), dim = 1))
    L3[0] = (self.input_encodings[2])(L3[0])
    (L3[0])[:, :segment_l3_s1_length, :] += self.segment_encodings_lvl_2_s1[0]
    (L3[0])[:, segment_l3_s1_length:, :] += self.segment_encodings_lvl_2_s1[1]
    L3[0] = (self.positional_encodings[2])(L3[0])
    L3[0] = (self.hace_blocks[2])(L3[0])

    L3.append(torch.cat(([L2[2], L2[3], L2[4]]), dim = 1))
    L3[1] = (self.input_encodings[3])(L3[1])
    (L3[1])[:, :segment_l3_s2_length, :] += self.segment_encodings_lvl_2_s2[0]
    (L3[1])[:, segment_l3_s2_length:2*segment_l3_s2_length, :] += self.segment_encodings_lvl_2_s2[1]
    (L3[1])[:, 2*segment_l3_s2_length:, :] += self.segment_encodings_lvl_2_s2[2]
    L3[1] = (self.positional_encodings[3])(L3[1])
    L3[1] = (self.hace_blocks[3])(L3[1])

    L3.append(torch.cat((L2[5], L2[6]), dim = 1))
    L3[2] = (self.input_encodings[2])(L3[2])
    (L3[2])[:, :segment_l3_s1_length, :] += self.segment_encodings_lvl_2_s1[0]
    (L3[2])[:, segment_l3_s1_length:, :] += self.segment_encodings_lvl_2_s1[1]
    L3[2] = (self.positional_encodings[2])(L3[2])
    L3[2] = (self.hace_blocks[2](L3[2]))

  #  print("Processing level 4")

    L4_pre = L3
    segment_l4_length = 28

    L4_pre[0] = (self.input_encodings[4])(L4_pre[0])
    L4_pre[1] = (self.input_encodings[5])(L4_pre[1])
    L4_pre[2] = (self.input_encodings[4])(L4_pre[2])

    L4 = [torch.cat((L4_pre[0], L4_pre[1], L4_pre[2]), dim = 1)]
    (L4[0])[:, :segment_l4_length, :] += self.segment_encodings_lvl_3[0]
    (L4[0])[:, segment_l4_length:2*segment_l4_length, :] += self.segment_encodings_lvl_3[1]
    (L4[0])[:, 2*segment_l4_length:, :] += self.segment_encodings_lvl_3[2]
    L4[0] = (self.positional_encodings[4])(L4[0])
    L4[0] = (self.hace_blocks[4])(L4[0])


    postprocessing_1_output = self.postprocessing_linear_1(L4[0])
    postprocessing_2_output = self.postprocessing_linear_2(postprocessing_1_output)
    postprocessing_3_output = self.postprocessing_linear_3(postprocessing_2_output)
    postprocessing_squeezed = postprocessing_3_output.squeeze(-1)
    pre_sigmoid_output = self.pre_output_linear(postprocessing_squeezed)
    output_layer = self.output_sigmoid(pre_sigmoid_output)


class InputEncoding(nn.Module):
  def __init__(self, d_model_in, d_model_out):
    super(InputEncoding, self).__init__()

    self.d_model_in = d_model_in
    self.d_model_out = d_model_out

    self.linear_embedding_layer = nn.Sequential(
        nn.Linear(in_features = self.d_model_in, out_features = self.d_model_out * 2),
        nn.ReLU(),
        nn.Linear(in_features = self.d_model_out * 2, out_features = self.d_model_out),
    )

  def forward(self, x):
    embedding = self.linear_embedding_layer(x)
    return embedding


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len = 10000, log_factor = 10000):
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

class HACEBlock(nn.Module):
  def __init__(self, num_tokens_in, num_tokens_out, d_model_in, d_model_out):
    super(HACEBlock, self).__init__()

    self.num_tokens_in = num_tokens_in
    self.num_tokens_out = num_tokens_out
    self.vertical = num_tokens_in // num_tokens_out

    self.d_model_in = d_model_in
    self.d_model_out = d_model_out

    #print(f"HACEBlock, d_model_in: {d_model_in}")

    num_heads = 4
    self.mha = MultiHeadAttention(d_model_in, num_heads)
    self.norm1 = nn.LayerNorm(d_model_in)

    self.ffn = nn.Sequential(
        nn.Linear(in_features = d_model_in, out_features = 4 * d_model_in),
        nn.ReLU(),
        nn.Linear(in_features = 4 * d_model_in, out_features = d_model_in)
    )
    self.norm2 = nn.LayerNorm(d_model_in)
    self.conv_layer = ConvolutionLayer(d_model = self.d_model_in, vertical = self.vertical, num_features = self.d_model_out)
    self.norm3 = nn.LayerNorm(self.d_model_out)


  def forward(self, x):
    mha_out = self.mha(x)
    residual_addition_attention = x + mha_out
    norm_attention = self.norm1(residual_addition_attention)

    ffn_output = self.ffn(norm_attention)
    residual_addition_ffn = norm_attention + ffn_output
    encoder_output = self.norm2(residual_addition_ffn)

    condensed_encodings = self.conv_layer(encoder_output)
    normalized_condensed_encodings = self.norm3(condensed_encodings)

    return normalized_condensed_encodings

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    try:
      self.head_dim = d_model // num_heads # get dimension for each head
    except:
      print(d_model)

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
  #  print(f"kernel size: {self.kernel_size}")
    self.stride = (vertical, 1)

    self.conv_layer = nn.Conv2d(in_channels = 1, out_channels = self.num_features, kernel_size = self.kernel_size, stride = self.stride)

  def forward(self, x):

    batch_size = x.size(0)
    num_tokens = x.size(1)

  #  print(f"before 4d transform, x shape: {x.shape}")
    x = x.unsqueeze(0).permute(1,0,2,3)
  #  print(f"after 4d transform, x shape: {x.shape}")

    convolution_output = self.conv_layer(x)
  #  print(f"convolution output shape: {convolution_output.shape}")
    n_condensed_tokens = convolution_output.size(2)

    condensed_encoding = (convolution_output.permute(0,2,1,3)).view(batch_size, n_condensed_tokens, self.num_features)
  #  print(f"condensed_encoding expected shape: {(batch_size, n_condensed_tokens, self.num_features)}, actual shape : {condensed_encoding.shape}")

    return condensed_encoding
