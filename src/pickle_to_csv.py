import pickle
import csv
import sys
import utils

pickle_file_path = '../output/sensor_data.pkl'
csv_file_path = '../output/sensor_data.csv'

with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

data.to_csv(csv_file_path)

