import torch
import torch.nn as nn

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
# Training function
def train(model, train_loader, criterion, optimizer):
    model.train(True)
    running_loss = 0.0
    
    for i, batch in enumerate(train_loader):
        # separate the batch into inputs and labels
        x_batch, y_batch = batch[0], batch[1]
        
        # calculate the prediction
        y_prediction = model(x_batch)
        loss = criterion(y_prediction, y_batch)
        running_loss += loss.item()
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print the loss every 50 batches
        if i % 50 == 49:
            avg_loss = running_loss / 50
            print(f"Batch {i + 1} loss: {avg_loss:.10f}")
            running_loss = 0.0
            
    print()

# Testing function    
def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    
    for i, batch in enumerate(test_loader):
        # separate the batch into inputs and labels
        x_batch, y_batch = batch[0], batch[1]
        
        # calculate the prediction without calculating gradients
        with torch.no_grad():
            y_prediction = model(x_batch)
            loss = criterion(y_prediction, y_batch)
            running_loss += loss.item()
            
    avg_loss = running_loss / len(test_loader)
    
    print(f"Test loss: {avg_loss:.10f}")