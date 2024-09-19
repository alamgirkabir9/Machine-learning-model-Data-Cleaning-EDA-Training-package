from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def normalize_data(df, columns=None):
    if columns is None:
        columns = df.columns
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_labels(df, columns):
    encoder = LabelEncoder()
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df

def get_dummies(df, columns):
    return pd.get_dummies(df, columns=columns)

def categorical(df):
    return df.select_dtypes(include=object).columns
