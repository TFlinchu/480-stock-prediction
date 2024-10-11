import os
from read_data import process_csv_files
import pandas as pd

"""
This script is currently used to test the functionality of the read_data.py file.
The code here is temporary and should be replaced once we start work on the main script.
"""

if __name__ == "__main__":
    # Process the APPL CSV file and get the tensor data
    tensor_data = process_csv_files("AAPL")

    # Print some of the data for testing
    for column, tensor in tensor_data.items():
        print(f"Column: {column}")
        if column == 'Date':
            # Convert timestamps back to readable date format
            dates = pd.to_datetime(tensor.numpy() * 10**9)
            date_strings = dates.strftime('%Y-%m-%d').tolist()
            print(f"Data (first 5 entries): {date_strings[:5]}")
        else:
            print(f"Data (first 5 entries): {tensor[:5]}")
        print()