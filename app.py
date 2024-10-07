import pandas as pd
import glob
import os

os.chmod("./archive", 0o666)
data = pd.read_csv('./archive')
folder_path = './archive'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

data_list = []
for file in csv_files:
    df = pd.read_csv(file)
    data_list.append(df)