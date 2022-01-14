import json
import numpy as np
import matplotlib.pyplot as plt

# funcs stolen from nyu deep learning course
from res.sequential_tasks import pad_sequences, to_categorical
import pandas as pd



## load data 

def load_sim_data(train_data_file_path, test_data_file_path):
    train_data_file = open(train_data_file_path)
    train_data_sim = json.load(train_data_file)

    test_data_file = open(test_data_file_path)
    test_data_sim = json.load(test_data_file)
    
    return (train_data_sim, test_data_sim)


train_data_file_path = 'simulated_data/train_data_v1.0.json'
test_data_file_path = 'simulated_data/test_data_v1.0.json'

(train_data_sim, test_data_sim) = load_sim_data(train_data_file_path, test_data_file_path)