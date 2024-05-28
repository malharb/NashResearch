targeted disorders: MDD (Major Depressive Disorder)

Hyperparameters:
Learning rate: 0.001
Optimizer: ADAM
Epochs: 12

Training data:
Depressed samples: 73
Healthy samples: 110

Model information:
num_encoder_levels = 4, num_heads = 5, d_model_in_list = [32,32,(32,48),32], d_model_out_list = [10,20,(30,32),40], token_in_list = [1440, 288, (144, 216), 84], token_out_list = [144, 72, (24, 36), 84], num_metrics = 12, dropout = 0.05, batch_size = 64, token_length = 20160
