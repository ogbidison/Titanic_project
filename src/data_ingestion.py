import pandas as pd
from pathlib import Path
#from .config import PATHS

#import requests
#from io import StringIO


class DataLoader:
    def __init__(self, raw_data):
        self.raw_data= raw_data


    def load_csv (self):
        return pd.read_csv(self.raw_data)

    def save_processed (self, df: pd.DataFrame, file_path: str, filename, num_cols):
        processed_path = Path(file_path)
        processed_path.mkdir(parents = True, exist_ok = True)
        df = df.copy()
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors = 'coerce')
        df.to_csv(processed_path/filename, index = False)

import inspect
print(">>> USING:", inspect.getfile(DataLoader))