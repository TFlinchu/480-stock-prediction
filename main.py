import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

from data_handler import StockDataset, process_csv_files, prepare_dataframe_for_lstm
from LSTM import LSTM, train, test

if __name__ == "__main__":   
    # Timer for entirety of program
    program_start_time = time.perf_counter()
    
    # Multiple stocks to train on
    training_stocks = ["AAPL", "MSFT"]
    
    # Create the Model
    model = LSTM(input_size=1, hidden_size=2250, num_layers=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Define important parameters
    learning_rate = 0.001
    num_epochs = 5
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get the data of the desired stock
    done = False
    while not done:
        stock = input("Enter the stock you would like to use: ")
        stock_data = process_csv_files(stock)

        if stock_data is None:
            print("The stock you entered is invalid. Please try again.")
        else:
            while True:
                ans = input("Would you like to train on multiple stocks? (y/n) ")
                if ans.lower() == "n": 
                    print("Testing on file " + stock + ".csv...")
                    done = True
                    stock_list = [stock]
                    break
                elif ans.lower() == "y":
                    print("Multiple stocks option chosen.")
                    done = True
                    stock_list = training_stocks + [stock]
                    break
                else: 
                    print("Invalid input. Please try again.")
                    
    # Loop through stock_list (1 or multiple)
    for stock in stock_list:
        stock_data = process_csv_files(stock)
        
        if stock_data is None:
            print(f"Training stock {stock} was invalid.")
            continue
        
        print(f"Training data for {stock} processed and prepared for LSTM.")
        
        # Convert 'Date' column to datetime format 
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], unit='s')
        
        # Prepare the data for the LSTM model
        lookback = 3
        shifted_data = prepare_dataframe_for_lstm(stock_data, lookback)    
        
        # Scale the data for the LSTM model
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
        
        # Sets training_start_time to a point in time
        training_start_time = time.perf_counter()
    
        # Train the model
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}:")
            train(model, train_loader, criterion, optimizer)
            test(model, test_loader, criterion)
            print("-----------------------------------------------")
            print()
            
        # Calculates training time by subtracting the current time by training_start_time
        elapsed_time = time.perf_counter() - training_start_time
        minutes = int(elapsed_time/60)
        seconds = int(elapsed_time%60)
        print(f"Training Time for {stock}: {minutes}:{seconds}")
        print()

    # Final test on the model
    print("Final Test for " + stock + ":")
    test(model, test_loader, criterion)
    
    # Inverse transform the entire scaled data
    scaled_data_inverse = scaler.inverse_transform(scaled_data)

    # Get the predictions for the testing set and converts it to a NumPy Array.
    test_predictions = model(x_test).detach().numpy().flatten()

    # Extract the actual close prices for the test set
    y_test_real = scaled_data_inverse[split_index:, 0]

    # Create an array with the same number of features as the original data
    test_predictions_full = np.zeros((test_predictions.shape[0], scaled_data.shape[1]))
    test_predictions_full[:, 0] = test_predictions

    # Inverse transform the predictions
    test_predictions_real = scaler.inverse_transform(test_predictions_full)[:, 0]
    
    # Calculates program time by subtracting the current time by program_start_time
    elapsed_time = time.perf_counter() - program_start_time
    minutes = int(elapsed_time/60)
    seconds = int(elapsed_time%60)
    print(f"Program Time: {minutes}:{seconds}")
    print()

    # Graph the testing set using Matplotlib
    # This will be much less accurate, as the model has not seen this data
    plt.plot(y_test_real, label='Actual Close')
    plt.plot(test_predictions_real, label='Predicted')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    