from Task_2.data_exploration import *

#%%
print('--------- load data ----------')
ratings = load_data()
ratings_small = create_small_practice_dataset(ratings)

#%%
print('---------- data exploration -------------')
data_exploration(ratings_small)

#%%
print('---------- pre-processing -----------')
