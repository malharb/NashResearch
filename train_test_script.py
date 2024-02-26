import torch
from torch import nn

def train_loop(dataloader, batch_size, model, lr = 0.001):

  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  loss_fn = nn.MSELoss()

  size = len(dataloader)
  model.train()

  for batch, (X,y) in enumerate(dataloader):

    model_screening = model(X)
    loss = loss_fn(model_screening, y)

    #backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 10 == 0:
      loss_val, current = loss.item(), (batch * batch_size) + len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model):

  loss_fn = nn.MSELoss()
  model.eval()

  num_batches = len(dataloader)
  test_loss = 0

  with torch.no_grad():
    for X,y in dataloader:
      model_screening = model(X)
      test_loss += loss_fn(model_screening, y).item()

  test_loss /= num_batches

  print(f"Avg loss: {test_loss:>8f} \n")

def train_and_test(model, train_dataloader, train_batch_size, test_dataloader, test_batch_size, epochs = 350):

  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, train_batch_size, model)
    test_loop(test_dataloader, model)

  print("Done.")
