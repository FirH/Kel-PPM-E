import pandas as pd
import numpy as np
def encode(df):
    # Map chirrosis stages to 1 and 0
    df['Stage'] = np.where(df['Stage'] == 4,1,0)
    
    # Categorical Bool Mapping
    df_subset = df[['Ascites','Hepatomegaly','Spiders', 'Sex', 'Drug']]
    df_subset = df_subset.stack().map({'Y':1,'N':0, 'F' : 1, 'M' : 0, 'D-penicillamine' : 0, 'Placebo' : 1}).unstack()

    # Edema & Status Mapping
    df_edema = df[['Edema', 'Status']]
    df_edema = df_edema.stack().map({'N':0,'S':1,'Y':2, 'C':0, 'D':1, 'CL':2}).unstack()

    # Df Merging
    df_bool_encoded = df.copy().drop('ID',axis =1)
    df_bool_encoded[['Ascites','Hepatomegaly','Spiders', 'Sex', 'Drug']] = df_subset[['Ascites','Hepatomegaly','Spiders', 'Sex', 'Drug']].astype('Int64')
    df_bool_encoded[['Edema', 'Status']] = df_edema[['Edema', 'Status']].astype('Int64')

    # Cast non-categorical int64 column to float64
    df_bool_encoded[['N_Days', 'Age']] = df_bool_encoded[['N_Days', 'Age']].astype('float64')
    
    # Drop rows with more than 5 NaN values
    df_bool_encoded.dropna(thresh=4, axis=1, inplace=True)
    
    return df_bool_encoded