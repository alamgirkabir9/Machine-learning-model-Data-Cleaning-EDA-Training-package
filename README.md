cleantransformer
cleantransformer is a Python package designed to simplify and automate common data cleaning, transformation, and exploratory data analysis (EDA) tasks. It also provides utilities for machine learning model training and evaluation, making it easier for data scientists and analysts to build and evaluate models.

Table of Contents
Introduction
Features
Installation
Usage
Data Cleaning
Data Transformation
Exploratory Data Analysis (EDA)
Model Training and Evaluation
Examples
Contributing
License
Introduction
cleantransformer provides a set of robust tools for data preprocessing, EDA, and model training. It aims to streamline the data science workflow by automating repetitive tasks and offering a standardized way to handle data preparation and analysis.

Features
Data Cleaning: Functions for handling missing values, removing duplicates, dropping columns/rows, and more.
Data Transformation: Utilities for normalizing, encoding, and creating dummy variables.
Exploratory Data Analysis (EDA): Functions for generating descriptive statistics and visualizations.
Model Training and Evaluation: Simplified interfaces for training and evaluating machine learning models.
Installation
To install the cleantransformer package, use the following command:

bash
Copy code
pip install cleantransformer
Usage
Data Cleaning
Use the data cleaning functions to preprocess your dataset:

python
Copy code
import pandas as pd
from cleantransformer.cleaning import handle_missing_values, remove_duplicates, drop_all_null_columns

# Load your dataset
df = pd.read_csv('path/to/your_dataset.csv')

# Remove columns with all null values
df = drop_all_null_columns(df)

# Remove duplicate rows
df = remove_duplicates(df)

# Handle missing values using the mean strategy
df = handle_missing_values(df, col=['column1', 'column2'], strategy='mean')
Data Transformation
Transform your dataset with normalization and encoding functions:

python
Copy code
from cleantransformer.transformation import normalize_data, encode_labels

# Normalize numerical columns
df = normalize_data(df, columns=['numerical_column1', 'numerical_column2'])

# Encode categorical columns
df = encode_labels(df, columns=['categorical_column'])
Exploratory Data Analysis (EDA)
Generate basic statistics and visualizations for your dataset:

python
Copy code
from cleantransformer.eda import generate_descriptive_stats, plot_correlation_matrix

# Print descriptive statistics
print(generate_descriptive_stats(df))

# Plot correlation matrix
plot_correlation_matrix(df)
Model Training and Evaluation
Train and evaluate machine learning models with ease:

python
Copy code
from cleantransformer.model import train_test_split_features_target, train_and_evaluate_classifiers,train_and_evaluate_regressor

# Split the dataset into features and target variable
X_train, X_test, y_train, y_test = train_test_split_features_target(df, target_column='target', test_size=0.3, random_state=42)

# Train and evaluate classifiers Model
best_model, train_acc, test_acc = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)
print(f"Best Model: {best_model}")
print(f"Training Accuracy: {train_acc}")
print(f"Testing Accuracy: {test_acc}")
# Train and evaluate Regressor Model
best_model, train_acc, test_acc = train_and_evaluate_regressor(X_train, X_test, y_train, y_test)
print(f"Best Model: {best_model}")
print(f"Training Accuracy: {train_acc}")
print(f"Testing Accuracy: {test_acc}")
