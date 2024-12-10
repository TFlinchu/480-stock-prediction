import glob
import os
import requests
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

def process_stock_listing():

    # An url to the api that will fetch our list of available stocks
    url ='https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo'
    
    # Read CSV file
    df = pd.read_csv(url)

    # Initialize a list to store column data
    column_data = []
    
    # Appends each value in the symbol column to list
    for value in df['symbol']:
        column_data.append(value)

    return column_data

def process_csv_files(stock):

    # An url to the api that will fetch our stock data
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + stock + '&outputsize=full&datatype=csv&apikey=E0WTCPU2OH7QNKY5'     

    # Read CSV file
    df = pd.read_csv(url)

    # Initialize a dictionary to store column data
    column_data = {}

    # Remove one of these columns from the list if we want to use it
    df = df.drop(columns=['open', 'high', 'low', 'volume'])
    
    # Convert 'timestamp' column to numeric format
    # This may not be necessary, as I can't imagine we would need to use the date in our model
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') / 10**9  # Convert to seconds since epoch
        
    # Reverse the order of the rows in the DataFrame
    df = df.iloc[::-1].reset_index(drop=True)

    # Store the data in the dictionary
    for column in df.columns:
        if column not in column_data:
            column_data[column] = []
        column_data[column].extend(df[column].tolist())

    # Convert lists in the dictionary to PyTorch tensors
    tensor_data = {}
    for key, value in column_data.items():
        try:
            tensor_data[key] = torch.tensor(value, dtype=torch.float32)
        except ValueError:
            # Should only happen if date column is not converted to numeric format
            print(f"Skipping column {key} due to non-numeric data")

    return tensor_data

# Prepare the data for the LSTM model by shifting the data by the lookback value
def prepare_dataframe_for_lstm(stock_data, lookback):
    # Convert the dictionary to a DataFrame
    stock_dataframe = pd.DataFrame(stock_data)
    
    # Set the 'timestamp' column as the index
    stock_dataframe.set_index('timestamp', inplace=True)
    
    for i in range(1, lookback + 1):
        # Create a new column for each lookback value
        stock_dataframe[f'close(t-{i})'] = stock_dataframe['close'].shift(i)
    
    # As we are shifting the data, NaN values will be created. This drops those rows
    stock_dataframe.dropna(inplace=True)
    
    # Flip the DataFrame so that the most recent data is at the end of the DataFrame
    shifted_data = np.flip(stock_dataframe.to_numpy(), axis=1)
    
    return shifted_data

# Create a Dataset class to use with DataLoader
class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]