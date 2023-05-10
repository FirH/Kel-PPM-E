import numpy as np
import pandas as pd

def imputasi(df):
    cat_columns = df.select_dtypes(include=['floating']).columns
    num_columns = df.select_dtypes(include=['integer', 'object']).columns
    classCol = df.columns[-1]
    for col in cat_columns:
        df[col] = df[col].fillna(df.groupby(classCol)[col].transform(pd.Series.mode))
    for col in num_columns:
        df[col] = df[col].fillna(df.groupby(classCol)[col].transform(pd.Series.median))
    return df

def zscore(df):
  list_fitur = df.select_dtypes(include=['integer', 'object']).columns[:-1]
  for fitur in list_fitur:
    df[fitur] = (df[fitur] - df[fitur].mean())/(df[fitur].std())
  return df
