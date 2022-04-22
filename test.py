import pandas as pd
import numpy as np

player_data = pd.read_csv('data/player_summary_with_xG_salary_2021.csv')
columns_to_scale = ['EV_for_per_60', 'PP_for_per_60', 'PK_for_per_60', 'EV_against_per_60',
                    'PP_against_per_60', 'PK_against_per_60', 'EV_netXG_per_60']

for column in columns_to_scale:
    player_data[column] = np.where(player_data[column] == '$0', 0, player_data[column])
    player_data[column] = np.where(player_data[column] == np.inf, 0, player_data[column])
    player_data[column] = np.where(player_data[column] == -np.inf, 0, player_data[column])
    player_data[column] = player_data[column].fillna(0)
    player_data[column] = player_data[column].apply(pd.to_numeric)
    player_data[column] = np.where(player_data[column] > 100, 100, player_data[column])

print(player_data[columns_to_scale].min())
print(player_data[columns_to_scale].max())