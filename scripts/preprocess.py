import pandas as pd

class cleaning:
    def load_data(inputfile):
        df=pd.read_csv(inputfile)
        return df

    def check_missing_values(df):
        return df.isnull().sum()

    def check_duplicate_values(df):
        return df.duplicated().sum()

    def data_types(df):
        return df.dtypes
