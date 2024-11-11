from src.predict_model import predict_model
from src.split_data import split_data
from src.train_model import train_model

if __name__ == '__main__':
    filepath = 'data/dubai_properties_processed_encoded.csv'
    split_data(filepath)
    train_model()
    predict_model()
