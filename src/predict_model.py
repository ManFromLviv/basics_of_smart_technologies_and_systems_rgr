# predict_model.py

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Metric evaluation function (for test cases, if true values exist)
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return mae, mse, rmse


def predict_model():
    model = joblib.load('src/random_forest_model.pkl')

    new_input = pd.read_csv('data/new_input.csv')

    if 'Rent' in new_input.columns:
        species_true = new_input['Rent']
        new_input = new_input.drop('Rent', axis=1)

    # Передбачення моделі
    predictions = model.predict(new_input)

    # Зчитування моделі
    mae, mse, rmse = evaluate_model(species_true, predictions)
    print(f"Random Forest:")
    print(f"\tMAE: {mae}")
    print(f"\tMSE: {mse}")
    print(f"\tRMSE: {rmse}")


