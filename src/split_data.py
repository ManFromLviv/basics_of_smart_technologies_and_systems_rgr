import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path):
    data = pd.read_csv(file_path)

    # Дані було підготовлено під час аналізу та кодування (файл rgr.ipynb)

    # Розділення на train та new_input у співвідношенні 80:20
    train_data, new_input = train_test_split(data, test_size=0.2, random_state=42)

    # Збереження даних
    train_data.to_csv('data/train_split.csv', index=False)
    new_input.to_csv('data/new_input.csv', index=False)
    print('Дані були оброблені, закодовані, розділені та збережені!')