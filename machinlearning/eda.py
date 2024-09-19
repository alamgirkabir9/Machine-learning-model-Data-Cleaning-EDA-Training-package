
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def descriptive(df):
    return df.describe()

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.show()

def plot_histograms(df, columns=None):
    if columns is None:
        columns = df.columns
    df[columns].hist(bins=15, figsize=(15, 10))
    plt.show()

def plot_boxplots(df, columns=None):
    if columns is None:
        columns = df.columns
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.show()
def plot_missing_values(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def pairplot(df):
    sns.pairplot(df)
    plt.title('Pairwise Relationships')
    plt.show()