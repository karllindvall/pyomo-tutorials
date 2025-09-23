'Construct dataloader from input and output data and run trainng loop for a model.'

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def make_dataloader(X, Y, batch_size=32, shuffle=True) -> DataLoader:
    """Make the DataLoader object with input and output data (X and Y)."""
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, loader:DataLoader, epochs, lr=1e-3):
    """Perform training loop over a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        epoch_loss = running_loss / len(loader.dataset)
        history.append(epoch_loss)

    return history