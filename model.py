import numpy as np
import pandas as pd
import pickle

def _preprocess_data(data):
    arr = np.array(data, dtype=np.float64)
    query = arr.reshape(1,-1)
    return query

def load_model(path_to_model):
    model = None
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(data, model):
    variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    query = _preprocess_data(data)
    prediction = variety_mappings[model.predict(query)[0]]
    return prediction