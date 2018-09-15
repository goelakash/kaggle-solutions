import pandas as pd

def load_train_file():
    return pd.read_csv('data/train.csv')

def load_test_file():
    return pd.read_csv('data/test.csv')
