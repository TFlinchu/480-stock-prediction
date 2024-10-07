<<<<<<< HEAD
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
=======
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
>>>>>>> 763a8cf51e70577035c65b952e54ea713d8c8096
    data_list.append(df)