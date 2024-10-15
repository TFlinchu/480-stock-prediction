import os
from read_data import process_csv_files
import pandas as pd
import torch
import torch.nn as nn
import torch.optim

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        cell0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (hidden0.detach(), cell0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, labels in loader: 
        outputs = model(data) 
        loss = criterion(outputs, labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        __, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            __, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


if __name__ == "__main__":

    train_loader = process_csv_files("A")
    test_loader = process_csv_files("A")

    model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"MLP Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
