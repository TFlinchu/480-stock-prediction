import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

from data_handler import StockDataset, process_csv_files, prepare_dataframe_for_lstm, process_stock_listing
from LSTM import LSTM, train, test, predict_future

if __name__ == "__main__":   
    # Timer for entirety of program
    program_start_time = time.perf_counter()
    
    # Create the Model
    model = LSTM(input_size=1, hidden_size=2250, num_layers=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Define important parameters
    learning_rate = 0.001
    num_epochs = 5
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Creates a list of available stocks in our api
    available_stocks = process_stock_listing()

    # Get the data of the desired stock
    done = False
    while not done:
        stock = input("Enter the stock you would like to use: ").upper()

        if stock not in available_stocks:
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
                    
                    correct = False
                    while not correct:
                        correct = True
                        # Multiple stocks to train on
                        training_stocks = input("Enter your training stock(s) separated with a space: ").upper()
                        done = True
                        stock_list = training_stocks.split() + [stock]
                        
                        for stockToTest in stock_list:
                            if stockToTest not in available_stocks:
                                print(f"Training stock {stockToTest} was invalid.")
                                correct = False
                    break
                else: 
                    print("Invalid input. Please try again.")
                    
    # Loop through stock_list (1 or multiple)
    for stock in stock_list:
        try:
            stock_data = process_csv_files(stock)
        except Exception as e:
            print(f"An error occurred while processing the data for {stock}: {e}")
            print("If this error persists, we may have reached the API limit for today")
            exit()
        
        if stock_data is None:
            print(f"Training stock {stock} was invalid.")
            continue
        
        print(f"Training data for {stock} processed and prepared for LSTM.")
        
        # Convert 'Date' column to datetime format 
        stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'], unit='s')
        
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

    # Get the predictions for the training set without training, and converts it to a NumPy Array. This will be used for graphing
    # train_predictions = model(x_train).detach().numpy().flatten()

    # Inverse transform the entire scaled data
    # scaled_data_inverse = scaler.inverse_transform(scaled_data)

    # Extract the actual close prices for the test set
    # y_train_real = scaled_data_inverse[:split_index, 0]

    # Create an array with the same number of features as the original data
    # train_predictions_full = np.zeros((train_predictions.shape[0], scaled_data.shape[1]))
    # train_predictions_full[:, 0] = train_predictions

    # Inverse transform the predictions
    # train_predictions_real = scaler.inverse_transform(train_predictions_full)[:, 0]

    # Graph the training set using Matplotlib
    # # This will be very accurate, as the model has already seen this data
    # plt.plot(y_train_real, label='Actual Close')
    # plt.plot(train_predictions_real, label='Predicted')
    # plt.xlabel('Day')
    # plt.ylabel('Close Price')
    # plt.legend()
    # plt.show()
    
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
    print(f"Total Program Time: {minutes}:{seconds}")
    print()

    # Graph the testing set using Matplotlib
    # This will be much less accurate, as the model has not seen this data
    plt.plot(y_test_real, label='Actual Close')
    plt.plot(test_predictions_real, label='Predicted')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # Predict future stock prices
    print()
    print("NOTE: The more days you predict, the less accurate the predictions will be.")
    while True:
        future_steps = input("How many days would you like to predict?: ")  # Number of future days to predict
        try:
            future_steps = int(future_steps)
            if future_steps <= 0:
                print("Invalid input. Please enter a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            
    # Train on the testing data once, just so we can predict after
    train(model, test_loader, criterion, optimizer)
            
    last_data = x_test[-1].unsqueeze(0)  # Use the last known data point
    future_predictions = predict_future(model, last_data, future_steps)

    # Inverse transform the future predictions
    future_predictions_full = np.zeros((future_steps, scaled_data.shape[1]))
    future_predictions_full[:, 0] = future_predictions
    future_predictions_real = scaler.inverse_transform(future_predictions_full)[:, 0]

    # Graph the future predictions
    plt.plot(range(future_steps), future_predictions_real, label='Future Predicted')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
