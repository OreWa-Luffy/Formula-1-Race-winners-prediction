import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    return pd.read_csv(filepath)


def inspect_data(dp):
    print(dp.head())
    print(dp.describe())
    print(dp.isnull().sum())
    print(dp.info())


# Uses a label encoder for getting average race pos.
def preprocess_data(dp):
    dp = dp.dropna().reset_index(drop=True)

    label_encoder = LabelEncoder()
    dp['Driver'] = label_encoder.fit_transform(dp['Driver'])
    dp['Position'] = label_encoder.fit_transform(dp['Position'])
    dp['Track'] = label_encoder.fit_transform(dp['Track'])

    # Average position on all races combined

    dp['Avg_Finish'] = dp.groupby('Driver')['Position'].transform('mean')
    return dp


def select_features(dp):
    features = dp[['Driver', 'Avg_Finish', 'Track']]
    target = dp['Position']
    return features, target


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - predictions))
    print(f'Mean squared Error: {mse}')
    return predictions, mse, rmse, mae


def tune_hyperparameters(X_train, y_train):
    model = GradientBoostingRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)
    print(f'Best parameters: {grid_search.best_params_}')
    best_model = grid_search.best_estimator_
    return best_model


def cross_validate_model(model, features, target):
    scores = cross_val_score(model, features, target, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    print(f'Cross validated mean Squared error: {mean_mse}')
    return mean_mse


def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel('Actual Position')
    plt.ylabel('Predicted Position')
    plt.title('Actual vs Predicted Position')
    plt.show()


def main():
    filepath = "archive/Formula1_2022-2024season_raceResults.csv"
    dp = load_data(filepath)
    inspect_data(dp)
    dp = preprocess_data(dp)
    features, target = select_features(dp)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    best_model = tune_hyperparameters(X_train_scaled, y_train)
    predictions, mse, rmse, mae = evaluate_model(best_model, X_test_scaled, y_test)
    visualize_results(y_test, predictions)


if __name__ == "__main__":
    main()
