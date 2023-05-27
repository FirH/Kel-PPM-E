import pandas as pd
import numpy as np

def oh_encode(df):
    # Map chirrosis stages to 1 and 0
    df['Stage'] = np.where(df['Stage'] == 4,1,0)
    
    # Dropping ID and rows with more than 5 NaN values
    df = df.drop('ID',axis =1) 
    df.dropna(thresh=5, axis=1, inplace=True)
    
    # One Hot Encoding
    cat_column = ['Ascites','Hepatomegaly','Spiders', 'Sex', 'Drug', 'Edema', 'Status']
    df_oh_encoded = pd.get_dummies(df, columns = cat_column)
    
    
    # Convert boolean OHE columns to float
    for columns in df_oh_encoded:
        if df_oh_encoded[columns].dtype == 'bool':
            
            df_oh_encoded[columns] = df_oh_encoded[columns].astype(int)
    
    # Moving Stage to last column
    ohe_columns = df_oh_encoded.columns.tolist()
    index = df_oh_encoded.columns.get_loc('Stage')
    new_index = ohe_columns[0:index] + ohe_columns[index+1:] + ohe_columns[index:index+1]
    df_oh_encoded = df_oh_encoded[new_index]
    
    # Cast non-categorical int64 column to float64
    df_oh_encoded[['N_Days', 'Age']] = df_oh_encoded[['N_Days', 'Age']].astype('float64')
    
    # df_oh_encoded[ohe_columns] = df_oh_encoded[ohe_columns].astype('Int64')
    
    
    return df_oh_encoded