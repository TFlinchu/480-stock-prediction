import glob
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

def process_csv_files(stock):
    # Path where the CSV files are stored, and set permissions
    folder_path = './archive'
    os.chmod(folder_path, 0o666)

    # Get the CSV files matching the stock chosen
    stock = stock + '.csv'
    csv_file_path = os.path.join(folder_path, stock)

    # Check if the .csv file exists
    if not os.path.isfile(csv_file_path):
        print(f"Error: The file {csv_file_path} does not exist.")
        return None
    else:
        csv_files = glob.glob(csv_file_path)

    # Initialize a dictionary to store column data
    column_data = {}

    # Read each CSV file, drop all columns but date and close (all we need to start), and store the data in the dictionary
    for file in csv_files:
        df = pd.read_csv(file)
        # Remove one of these columns from the list if we want to use it
        df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Stock Splits', 'Dividends'])
        
        # Convert 'Date' column to numeric format
        # This may not be necessary, as I can't imagine we would need to use the date in our model
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).astype('int64') / 10**9  # Convert to seconds since epoch

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
    
    # Set the 'Date' column as the index
    stock_dataframe.set_index('Date', inplace=True)
    
    for i in range(1, lookback + 1):
        # Create a new column for each lookback value
        stock_dataframe[f'Close(t-{i})'] = stock_dataframe['Close'].shift(i)
    
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