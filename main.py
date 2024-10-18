import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_handler import StockDataset, process_csv_files, prepare_dataframe_for_lstm
from LSTM import LSTM, train, test

if __name__ == "__main__":
    # Get the data of the desired stock
    stock = 'AAPL'
    stock_data = process_csv_files(stock)
    
    # Convert 'Date' column to datetime format 
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], unit='s')
    
    # Define how far back we want to consider, and shift the data
    lookback = 7  
    shifted_data = prepare_dataframe_for_lstm(stock_data, lookback)

    # Scale the data for the LSTM model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(shifted_data)
    
    # Separate the data into inputs and labels
    x = scaled_data[:, 1:]
    y = scaled_data[:, 0]
    
    # Split the data into training and testing sets. 95% training, 5% testing
    split_index = int(0.95 * len(x))
    
    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    # Define that each input has one feature
    x_train = x_train.reshape(-1, lookback, 1)
    x_test = x_test.reshape(-1, lookback, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Convert the data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Create the Datasets and DataLoaders
    train_dataset = StockDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    test_dataset = StockDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create the Model
    model = LSTM(input_size=1, hidden_size=4, num_layers=1)
    
    # Define important parameters
    learning_rate = 0.001
    num_epochs = 15
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}:")
        train(model, train_loader, criterion, optimizer)
        test(model, test_loader, criterion)
        print("-----------------------------------------------")
        print()
        
    # Final test on the model
    print("Final Test:")
    test(model, test_loader, criterion)