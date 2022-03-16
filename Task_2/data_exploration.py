import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_data():
    ratings = pd.read_csv('data/ratings.csv')
    return ratings

def create_small_practice_dataset(ratings):
    selected_ids = np.random.choice(ratings['userId'].unique(), size=int(len(ratings['userId'].unique())*0.2))
    ratings_small = ratings.loc[ratings['userId'].isin(selected_ids)]
    return ratings_small

def data_exploration(ratings_small):
    print('data selected from ', len(ratings_small['userId'].unique()), ' unique users')
    # add more data exploration dx
    return
