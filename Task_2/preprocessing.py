# imports
import pandas as pd

def train_test_split(ratings):
    # order per user the ratings s.t. timestamp is descending, then newest_rated = 1 for the newest ranked
    ratings['newest_rated'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

    # make train test split
    train = ratings[ratings['newest_rated'] != 1]
    test = ratings[ratings['newest_rated'] == 1]

    # drop time step column because we no longer need it
    train = train[['userId', 'movieId', 'rating']]
    test = test[['userId', 'movieId', 'rating']]

    return train, test

# TODO:
# - one-hot-encode user id's
# - transform to implicit data
# - train-test split
