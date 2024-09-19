import pandas as pd
from machinlearning.cleaning import drop_all_null_columns,drop_columns,remove_duplicates
from machinlearning.transformation import encode_labels,categorical
from machinlearning.model import train_test_split_features,train_and_evaluate_classifiers
pd.set_option('display.max_columns', None)
df = pd.read_csv('F:/cleantransformer_package/Churn_Modelling.csv') 
print(df.columns)
df=drop_all_null_columns(df)
df=drop_columns(df,columns=['RowNumber', 'CustomerId','Surname'])
df=remove_duplicates(df)
print(df.head())
columns = categorical(df)
df=encode_labels(df,columns)
X_train, X_test, y_train, y_test=train_test_split_features(df, target_column='Exited', test_size=0.3, random_state=42)

print("Training Features (X_train):")
print(X_train)
print("\nTesting Features (X_test):")
print(X_test)
print("\nTraining Target (y_train):")
print(y_train)
print("\nTesting Target (y_test):")
print(y_test)

model_name,train_accuracy, test_accuracy,model = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)
print(f"Best model:{model_name}")
print(f"Training Accuracy (R² score): {train_accuracy:.4f}")
print(f"Testing Accuracy (R² score): {test_accuracy:.4f}")
