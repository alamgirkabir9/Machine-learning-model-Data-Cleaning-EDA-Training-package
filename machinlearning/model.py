from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd

def train_test_split_features(df, target_column, test_size, random_state):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_regressor(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor()
    }

    best_model_info = {
        'model_name': None,
        'best_model': None,
        'train_accuracy': -float('inf'),
        'test_accuracy': -float('inf')
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = r2_score(y_train, y_train_pred)
        test_accuracy = r2_score(y_test, y_test_pred)

        if test_accuracy > best_model_info['test_accuracy']:
            best_model_info['model_name'] = model_name
            best_model_info['best_model'] = model
            best_model_info['train_accuracy'] = train_accuracy
            best_model_info['test_accuracy'] = test_accuracy

    return best_model_info['model_name'], best_model_info['train_accuracy'], best_model_info['test_accuracy'], best_model_info['best_model']

def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, solver='lbfgs'),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier()
    }

    best_model_info = {
        'model_name': None,
        'best_model': None,
        'train_accuracy': -float('inf'),
        'test_accuracy': -float('inf')
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            if test_accuracy > best_model_info['test_accuracy']:
                best_model_info['model_name'] = model_name
                best_model_info['best_model'] = model
                best_model_info['train_accuracy'] = train_accuracy
                best_model_info['test_accuracy'] = test_accuracy

    return best_model_info['model_name'], best_model_info['train_accuracy'], best_model_info['test_accuracy'], best_model_info['best_model']
