from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib
import numpy as np
import pandas as pd

min_max_scaler = MinMaxScaler()

def identify_nn(player_data, player_name, number_nn):

    # Columns used to find nn
    columns_to_scale = ['EV_TOI_seconds', 'PP_TOI_seconds', 'SH_TOI_seconds', 'EV_for_agg',
                        'PP_for_agg', 'PK_for_agg', 'EV_against_agg', 'PP_against_agg',
                        'PK_against_agg', 'Shots', 'Goals', 'Assists', 'Hits',
                        'PIM', 'EV_netXG', 'EV_netXG_per_60']

    # Scaled df
    scaled_df = pd.DataFrame(min_max_scaler.fit_transform(player_data[columns_to_scale]))

    # Find nearest neighbords
    knn = NearestNeighbors(n_neighbors=number_nn)
    knn.fit(scaled_df)
    distance_mat, neighbours_mat = knn.kneighbors(scaled_df)

    # Find index for player
    index_player = player_data[player_data['Player']==player_name].index.values[0]

    # Return nearest_neighbors
    nearest_neighbors = player_data.iloc[neighbours_mat[index_player]]

    # Filter only for same position as desired player
    player_pos = nearest_neighbors.iloc[0]['position']
    if player_pos =='D':
        nearest_neighbors = nearest_neighbors[nearest_neighbors['position']=='D']
    else:
        nearest_neighbors = nearest_neighbors[nearest_neighbors['position'] != 'D']

    nearest_neighbors = nearest_neighbors[nearest_neighbors['num_salary']>0]
    current_salary = round(nearest_neighbors[:1].num_salary / 1000000, 2)
    fair_salary = round(np.mean(nearest_neighbors[1:].num_salary) / 1000000, 2)

    print(str('Nearest neighbor for :' + str(player_name)))
    print(str('Current salary (M) :' + str(current_salary)))
    print(str('Fair salary (M) :' + str(fair_salary)))
    print('----------------------------------')
    print(nearest_neighbors)

    return nearest_neighbors



