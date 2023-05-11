import numpy as np
import pandas as pd

def impute(df):
    num_columns = df.select_dtypes(include=['floating']).columns
    cat_columns = df.select_dtypes(include=['integer', 'object']).columns
    classCol = df.columns[-1]
    for col in cat_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in num_columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def zscore(df):
  list_fitur = df.select_dtypes(include=['integer', 'object']).columns[:-1]
  for fitur in list_fitur:
    df[fitur] = (df[fitur] - df[fitur].mean())/(df[fitur].std())
  return df
