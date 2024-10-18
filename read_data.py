import glob
import os
import pandas as pd
import torch

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