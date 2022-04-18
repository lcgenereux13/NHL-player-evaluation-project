import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

from src.plot_game_summary import plot_game_summary
from src.evaluate_pairings import create_summary
from src.identify_nearest_neighbors import identify_nn
import warnings
warnings.filterwarnings("ignore")

################################################################################################
# Plotting a game trajectory
################################################################################################

# plot_game_summary(20212022, 1061)

################################################################################################
# Extracting top players
################################################################################################

player_summary = pd.read_csv('data/player_summary_with_xG_salary_2021.csv')
player_summary_desired_columns = player_summary[['Player','cap_hit', 'GP', 'Goals',
                                                 'Assists','EV_netXG_per_60', 'clutch_factor']]
player_summary_desired_columns = player_summary_desired_columns.sort_values(by='EV_netXG_per_60', ascending=False)
print(player_summary_desired_columns[player_summary_desired_columns['GP']>40][0:50])


################################################################################################
# Extracting top non rookie players
################################################################################################

player_summary = pd.read_csv('data/player_summary_with_xG_salary_2021.csv')
player_summary = player_summary[player_summary['num_salary']>950000]
player_summary_desired_columns = player_summary[['Player','cap_hit', 'GP', 'Goals',
                                                 'Assists','EV_netXG_per_60', 'clutch_factor', 'num_salary']]
player_summary_desired_columns['EV_netXG_per_60_per_dollar'] = player_summary_desired_columns['EV_netXG_per_60']/\
                                                                player_summary_desired_columns['num_salary']
player_summary_desired_columns = player_summary_desired_columns.sort_values(by='EV_netXG_per_60_per_dollar', ascending=False)
print(player_summary_desired_columns[player_summary_desired_columns['GP']>40][0:50])


################################################################################################
# Extracting top players for a select team
################################################################################################

player_summary = pd.read_csv('data/player_summary_with_xG_salary_2021.csv')
player_summary_subset = player_summary[(player_summary['team_abb']=='EDM') &
                                       (player_summary['GP']>20)]
print(player_summary_subset[['Player','cap_hit', 'GP', 'Goals',
                                                 'Assists','EV_netXG_per_60', 'clutch_factor']])

################################################################################################
# Evaluating player pairings
################################################################################################

# Importing data
game_summary_select_year = pd.read_csv('data/player_summary_with_xG_salary_2021.csv')
game_summary_select_year['player_name'] = game_summary_select_year['Player']
shots_df = pd.read_csv('data/shots_and_xG.csv')
shots_df = shots_df[shots_df['year']==20212022]

# Assign player to teams - ensures that players who played on two teams can be accounted for on both teams
game_sum = pd.read_csv('data/game_sums_with_XG_21.csv')
player_summary_df = game_sum[['player_name','team_abb',
                           'player_number', 'position']].groupby(['player_name',  'team_abb']).tail(1)
player_summary_df['team'] = player_summary_df['team_abb']
player_summary_df['player'] = player_summary_df['player_name']
player_summary_df['number'] = player_summary_df['player_number']
player_summary_df['pos'] = player_summary_df['position']

# print(player_summary_df)
# print(player_summary_df[player_summary_df['player_name'] =='TOFFOLI, TYLER'])

# Extract pairings data
agg_summary = create_summary(player_summary_df, shots_df, 'MTL', 'A')
print(agg_summary[agg_summary['reference_player']==17])

# Visualize (only for situations where players have at least 75 shots for together)
agg_summary_material = agg_summary[agg_summary['shots_with_ref']>75]
heatmap_data= pd.pivot_table(agg_summary_material, values='delta_with_ref', index=['reference_player'],
                columns=['peer'], aggfunc=[np.sum])
print(heatmap_data)
fig = plt.figure(figsize=(24, 12))
# sns.diverging_palette(10, 200, s=20)
sns.set(font_scale=1.4)
sns.heatmap(heatmap_data.values,
            xticklabels=heatmap_data.index.values, yticklabels=heatmap_data.index.values, cmap="Spectral")
plt.show()

################################################################################################
# Nearest neighbors
################################################################################################

player_data = pd.read_csv('data/player_summary_with_xG_salary_2021.csv')
columns_to_scale = ['EV_TOI_seconds', 'PP_TOI_seconds', 'SH_TOI_seconds', 'EV_for_agg',
                    'PP_for_agg', 'PK_for_agg', 'EV_against_agg', 'PP_against_agg',
                    'PK_against_agg', 'Shots', 'Goals', 'Assists', 'Hits',
                    'PIM', 'EV_netXG', 'EV_netXG_per_60']

identify_nn(player_data, 'SUZUKI, NICK', 10, columns_to_scale)
identify_nn(player_data, 'LEHKONEN, ARTTURI', 10, columns_to_scale)

columns_to_scale = ['EV_TOI_seconds', 'PP_TOI_seconds', 'SH_TOI_seconds', 'EV_for_agg',
                    'PP_for_agg', 'PK_for_agg', 'EV_against_agg', 'PP_against_agg',
                    'PK_against_agg', 'EV_netXG', 'EV_netXG_per_60']

identify_nn(player_data, 'SUZUKI, NICK', 10, columns_to_scale)
identify_nn(player_data, 'LEHKONEN, ARTTURI', 10, columns_to_scale)

columns_to_scale = ['EV_for_per_60', 'PP_for_per_60', 'PK_for_per_60', 'EV_against_per_60',
                    'PP_against_per_60', 'PK_against_per_60', 'EV_netXG_per_60']

identify_nn(player_data, 'SUZUKI, NICK', 15, columns_to_scale)
identify_nn(player_data, 'LEHKONEN, ARTTURI', 15, columns_to_scale)
