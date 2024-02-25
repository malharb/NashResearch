import torch
from torch import nn

class NashModel(nn.Module):

  def __init__(self, composition):

    super().__init__()

    self.dropout_toggle = composition["dropout_toggle"]
    self.dropout_p = composition["dropout_p"]

    self.lstm_n_features = 2
    self.lstm_units = composition["lstm_units"]
    self.n_denselayer_units_post_lstm = composition["n_dense_layers_post_lstm"]
    self.lstm_encoding_size = composition["lstm_encoding_size"]

    self.static_n_features = 7
    self.n_denselayer_units_post_static = composition["n_denselayer_units_post_static"]
    self.static_encoding_size = composition["static_encoding_size"]

    #DAY 1
    self.lstm1 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm1_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm1_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm1_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static1 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static1_dropout = nn.Dropout(p = self.dropout_p)
    self.static1_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )

    # DAY 2
    self.lstm2 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm2_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm2_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm2_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static2 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static2_dropout = nn.Dropout(p = self.dropout_p)
    self.static2_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )

    # DAY 3
    self.lstm3 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm3_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm3_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm3_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static3 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static3_dropout = nn.Dropout(p = self.dropout_p)
    self.static3_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )

    # DAY 4
    self.lstm4 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm4_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm4_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm4_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static4 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static4_dropout = nn.Dropout(p = self.dropout_p)
    self.static4_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )
  
    # DAY 5
    self.lstm5 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm5_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm5_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm5_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static5 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static5_dropout = nn.Dropout(p = self.dropout_p)
    self.static5_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )

    # DAY 6
    self.lstm6 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm6_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm6_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm6_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static6 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static6_dropout = nn.Dropout(p = self.dropout_p)
    self.static6_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )

    # DAY 7
    self.lstm7 = nn.LSTM(input_size = 2, hidden_size = self.lstm_units, batch_first = True)
    self.post_lstm7_stack = nn.Sequential(
        nn.Linear(in_features = self.lstm_units, out_features = self.n_denselayer_units_post_lstm[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[0], out_features = self.n_denselayer_units_post_lstm[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[1], out_features = self.n_denselayer_units_post_lstm[2]),
        nn.ReLU(),
    )
    self.post_lstm7_dropout = nn.Dropout(p = self.dropout_p)
    self.lstm7_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_lstm[2], out_features = self.lstm_encoding_size),
        nn.ReLU()
    )

    self.static7 = nn.Sequential(
        nn.Linear(in_features = self.static_n_features, out_features = self.n_denselayer_units_post_static[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[0], out_features = self.n_denselayer_units_post_static[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_post_static[1], out_features = self.n_denselayer_units_post_static[2]),
        nn.ReLU(),
    )
    self.post_static7_dropout = nn.Dropout(p = self.dropout_p)
    self.static7_encoding = nn.Sequential(
        nn.Linear(in_features = self.n_denselayer_units_post_static[2], out_features = self.static_encoding_size),
        nn.ReLU(),
    )
 
    # Second Stack
    self.n_secondstack_lstm_features = self.lstm_encoding_size + self.static_encoding_size
    self.secondstack_lstm_units = composition["secondstack_lstm_units"]
    self.secondstack_lstm = nn.LSTM(input_size = self.n_secondstack_lstm_features, hidden_size = self.secondstack_lstm_units, batch_first = True)
    self.secondstack_processing = nn.Sequential(
        nn.Linear(in_features = self.secondstack_lstm_units, out_features = self.n_denselayer_units_secondstack[0]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_secondstack[0], out_features = self.n_denselayer_units_secondstack[1]),
        nn.ReLU(),
        nn.Linear(in_features = self.n_denselayer_units_secondstack[1], out_features = self.n_denselayer_units_secondstack[2]),
        nn.ReLU(),
    )

    self.output_layer = nn.Sequential(
      nn.Linear(in_features = self.n_denselayer_units_secondstack[2], out_features = 1),
      nn.ReLU(),
    )

  def forward(self, X):

    (lstm1_input, static1_input) = (X[0], X[1])
    (lstm2_input, static2_input) = (X[2], X[3])
    (lstm3_input, static3_input) = (X[4], X[5])
    (lstm4_input, static4_input) = (X[6], X[7])
    (lstm5_input, static5_input) = (X[8], X[9])
    (lstm6_input, static6_input) = (X[10], X[11])
    (lstm7_input, static7_input) = (X[12], X[13])

    #day 1
    lstm1_output = (self.lstm1(lstm1_input))[1][0][0]
    post_lstm1_stack_output = (self.post_lstm1_stack(lstm1_output))
    post_lstm1_dropout_output = (self.post_lstm1_dropout(post_lstm1_stack_output))
    lstm1_encoding_output = (self.lstm1_encoding(post_lstm1_dropout_output))

    static1_output = self.static1(static1_input)
    post_static1_dropout_output = self.post_static1_dropout(static1_output)
    static1_encoding_output = self.static1_encoding(post_static1_dropout_output)

    day1_encoding = torch.cat([lstm1_encoding_output, static1_encoding_output], dim = 1) #shape (batches, encoding size)

    #day2
    lstm2_output = (self.lstm2(lstm2_input))[1][0][0]
    post_lstm2_stack_output = (self.post_lstm2_stack(lstm2_output))
    post_lstm2_dropout_output = (self.post_lstm2_dropout(post_lstm2_stack_output))
    lstm2_encoding_output = (self.lstm2_encoding(post_lstm2_dropout_output))

    static2_output = self.static2(static2_input)
    post_static2_dropout_output = self.post_static2_dropout(static2_output)
    static2_encoding_output = self.static2_encoding(post_static2_dropout_output)

    day2_encoding = torch.cat([lstm2_encoding_output, static2_encoding_output], dim = 1) #shape (batches, encoding size)

    #day3
    lstm3_output = (self.lstm3(lstm3_input))[1][0][0]
    post_lstm3_stack_output = (self.post_lstm3_stack(lstm3_output))
    post_lstm3_dropout_output = (self.post_lstm3_dropout(post_lstm3_stack_output))
    lstm3_encoding_output = (self.lstm3_encoding(post_lstm3_dropout_output))

    static3_output = self.static3(static3_input)
    post_static3_dropout_output = self.post_static3_dropout(static3_output)
    static3_encoding_output = self.static3_encoding(post_static3_dropout_output)

    day3_encoding = torch.cat([lstm3_encoding_output, static3_encoding_output], dim = 3) #shape (batches, encoding size)

    #day4
    lstm4_output = (self.lstm4(lstm4_input))[1][0][0]
    post_lstm4_stack_output = (self.post_lstm4_stack(lstm4_output))
    post_lstm4_dropout_output = (self.post_lstm4_dropout(post_lstm4_stack_output))
    lstm4_encoding_output = (self.lstm4_encoding(post_lstm4_dropout_output))

    static4_output = self.static4(static4_input)
    post_static4_dropout_output = self.post_static4_dropout(static4_output)
    static4_encoding_output = self.static4_encoding(post_static4_dropout_output)

    day4_encoding = torch.cat([lstm4_encoding_output, static4_encoding_output], dim = 4) #shape (batches, encoding size)

    #day5
    lstm5_output = (self.lstm5(lstm5_input))[1][0][0]
    post_lstm5_stack_output = (self.post_lstm5_stack(lstm5_output))
    post_lstm5_dropout_output = (self.post_lstm5_dropout(post_lstm5_stack_output))
    lstm5_encoding_output = (self.lstm5_encoding(post_lstm5_dropout_output))

    static5_output = self.static5(static5_input)
    post_static5_dropout_output = self.post_static5_dropout(static5_output)
    static5_encoding_output = self.static5_encoding(post_static5_dropout_output)

    day5_encoding = torch.cat([lstm5_encoding_output, static5_encoding_output], dim = 5) #shape (batches, encoding size)

    #day6
    lstm6_output = (self.lstm6(lstm6_input))[1][0][0]
    post_lstm6_stack_output = (self.post_lstm6_stack(lstm6_output))
    post_lstm6_dropout_output = (self.post_lstm6_dropout(post_lstm6_stack_output))
    lstm6_encoding_output = (self.lstm6_encoding(post_lstm6_dropout_output))

    static6_output = self.static6(static6_input)
    post_static6_dropout_output = self.post_static6_dropout(static6_output)
    static6_encoding_output = self.static6_encoding(post_static6_dropout_output)

    day6_encoding = torch.cat([lstm6_encoding_output, static6_encoding_output], dim = 6) #shape (batches, encoding size)

    #day7
    lstm7_output = (self.lstm7(lstm7_input))[1][0][0]
    post_lstm7_stack_output = (self.post_lstm7_stack(lstm7_output))
    post_lstm7_dropout_output = (self.post_lstm7_dropout(post_lstm7_stack_output))
    lstm7_encoding_output = (self.lstm7_encoding(post_lstm7_dropout_output))

    static7_output = self.static7(static7_input)
    post_static7_dropout_output = self.post_static7_dropout(static7_output)
    static7_encoding_output = self.static7_encoding(post_static7_dropout_output)

    day7_encoding = torch.cat([lstm7_encoding_output, static7_encoding_output], dim = 7) #shape (batches, encoding size)

    #concatenate all day encodings 
    day_encodings = torch.cat([day1_encoding.unsqueeze(1), day2_encoding.unsqueeze(1), day3_encoding.unsqueeze(1), day4_encoding.unsqueeze(1),
                               day5_encoding.unsqueeze(1), day6_encoding.unsqueeze(1), day7_encoding.unsqueeze(1)], dim = 1)

    secondstack_lstm_output = (self.secondstack_lstm(day_encodings))[1][0][0]
    secondstack_processing_output = self.secondstack_processing(secondstack_lstm_output)

    #output
    nn_output = self.output_layer(secondstack_processing_output)

    return nn_output
