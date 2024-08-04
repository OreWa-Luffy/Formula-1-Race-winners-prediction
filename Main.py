import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    return pd.read_csv(filepath)


def inspect_data(df):
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    print(df.info())


# Uses a label encoder for getting average race pos.
def preprocess_data(dp):
    dp = dp.dropna()
    label_encoder = LabelEncoder()
    dp['Race'] = label_encoder.fit_transform(dp['Race'])
    dp['Driver'] = label_encoder.fit_transform(dp['Driver'])
    dp['Position'] = label_encoder.fit_transform(dp['Position'])
    dp['Track'] = label_encoder.fit_transform(dp['Track'])

    # Average position on all races combined

    dp['Avg_Finish'] = dp.groupby('Driver')['Position'].transform('mean')
    return dp


def select_features(dp):
    features = dp[['Race', 'Driver', 'Avg_Finish', 'Track']]
    target = dp['Position']
    return features, target


def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean squared Error: {mse}')
    return predictions, mse


def tune_hyperparameters(X_train, y_train):
    model = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f'Best parameters: {grid_search.best_params_}')
    best_model = grid_search.best_estimator_
    return best_model


def visualize_results(y_test,predictions):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel('Actual Position')
    plt.ylabel('Predicted Position')
    plt.title('Actual vs Predicted Position')
    plt.show()


def main():
    filepath = "archive/Formula1_2022-2024season_raceResults.csv"

