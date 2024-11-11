# train_model.py

import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_model():
    train_data = pd.read_csv('data/train_split.csv')

    # Розподіл даних на ознаки і цільову змінну
    X_train = train_data.drop('Rent', axis=1)
    y_train = train_data['Rent']


    # Налаштування гіперпараметрів
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor()

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=5, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

    # Тренування моделі
    random_search.fit(X_train, y_train)

    # Збереження найркащої моделі
    best_model = random_search.best_estimator_
    joblib.dump(best_model, 'src/random_forest_model.pkl')
    print("Модель збережено!")

