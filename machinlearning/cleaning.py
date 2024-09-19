import pandas as pd

def drop_all_null_columns(df):
    df = df.dropna(axis=1, how='all')
    df = df.reset_index(drop=True)
    return df

def drop_columns(df, columns):
    df = df.drop(columns=columns)
    df = df.reset_index(drop=True)
    return df

def drop_rows_nulls(df):
    df = df.dropna(how='any')
    df = df.reset_index(drop=True)
    return df

def handle_missing_values(df, col, strategy):
    columns = col
    for column in columns:
        if df[column].dtype == 'object':  # Categorical data
            df[column] = df[column].fillna(df[column].mode()[0])
        else:  # Numerical data
            if strategy == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'mode':
                df[column] = df[column].fillna(df[column].mode()[0])
    return df

def replace_columns(df, columns, target, value):
    for column in columns:
        df[column] = df[column].replace(target, value)
    return df

def remove_duplicates(df):
    return df.drop_duplicates()
