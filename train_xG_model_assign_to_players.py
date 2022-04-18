from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import requests
import re
import os
import random
import time
import regex
import json
import math

from decimal import Decimal

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn
from sklearn.inspection import plot_partial_dependence

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

import pickle
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")


############################################################################################
# Import 3 years of play by play
############################################################################################

# train on 3 years of data
play_by_play = pd.read_csv('data/play_by_play_2019.csv')
play_by_play = play_by_play.append(pd.read_csv('data/play_by_play_2020.csv'))
play_by_play = play_by_play.append(pd.read_csv('data/play_by_play_2021.csv'))

# Extract team name
play_by_play['year'] = list(map(lambda x: x.split('-')[0], play_by_play.year_game_home_away))
play_by_play['game'] = list(map(lambda x: x.split('-')[1], play_by_play.year_game_home_away))
play_by_play['team_1'] = list(map(lambda x: x.split('-')[2].replace('É', 'E'), play_by_play.year_game_home_away))
play_by_play['team_2'] = list(map(lambda x: x.split('-')[3].replace('É', 'E'), play_by_play.year_game_home_away))

# Create clean game
play_by_play['clean_game'] = play_by_play['year'].astype(str) +'-' + play_by_play['game'].astype(str) +  '-' + \
                                 play_by_play['team_1'].astype(str) +  '-' + play_by_play['team_2'].astype(str)

# Create ID
play_by_play['full_play_id'] = play_by_play['clean_game'] +  '-' +  \
                                  play_by_play['Period'].astype(str) +  '-' + play_by_play['Start'].astype(str)

print('Play by play data read in')

############################################################################################
# Import 3 years of X-Y coordinates
############################################################################################

x_y_coord = pd.read_csv('data/play_by_play_coord_2019.csv')
x_y_coord = x_y_coord.append(pd.read_csv('data/play_by_play_coord_2020.csv'))
x_y_coord = x_y_coord.append(pd.read_csv('data/play_by_play_coord_2021.csv'))
x_y_coord = x_y_coord.reset_index()

# Extract team name
x_y_coord['year'] = list(map(lambda x: x.split('-')[0], x_y_coord.year_game_home_away))
x_y_coord['game'] = list(map(lambda x: x.split('-')[1], x_y_coord.year_game_home_away))
x_y_coord['team_1'] = list(map(lambda x: x.split('-')[2].replace('É', 'E'), x_y_coord.year_game_home_away))
x_y_coord['team_2'] = list(map(lambda x: x.split('-')[3].replace('É', 'E'), x_y_coord.year_game_home_away))

# Create clean game
x_y_coord['clean_game'] = x_y_coord['year'].astype(str) +'-' + x_y_coord['game'].astype(str) +  '-' + \
                          x_y_coord['team_1'].astype(str) +  '-' + x_y_coord['team_2'].astype(str)

# Create ID
x_y_coord['full_play_id'] = x_y_coord['clean_game'] +  '-' +  \
                            x_y_coord['period'].astype(str) +  '-' + x_y_coord['periodTime'].astype(str)

# Select only key columns and drop duplicates (to ensure that only one location is taken if there are multiple at one time)
x_y_coord = x_y_coord[['full_play_id', 'x_coord', 'y_coord']].drop_duplicates()
x_y_coord = x_y_coord.groupby('full_play_id').head(1)

print('X-Y coordinates data read in')

############################################################################################
# Clean shots data
############################################################################################

# Merge shots and coordinates
merged = play_by_play.merge(x_y_coord, how='left', on='full_play_id')

shots = merged[(merged['Description'] == 'SHOT') |
              #(merged['Description'] == 'BLOCK') |
              (merged['Description'] == 'MISS') |
              (merged['Description'] == 'GOAL')].fillna(0)

shots['team_for'] = list(map(lambda x: x[0:3], shots['Details']))

# Correcting for possible double entries
shots['team_for'] = np.where(shots['team_for']=='L.A', 'LAK', shots['team_for'])
shots['team_for'] = np.where(shots['team_for']=='N.J', 'NJD', shots['team_for'])
shots['team_for'] = np.where(shots['team_for']=='S.J', 'SJS', shots['team_for'])
shots['team_for'] = np.where(shots['team_for']=='T.B', 'TBL', shots['team_for'])


############################################################################################
# Identify home and away players on ice for each shot
############################################################################################

# Create array of home team positions and numbers on ice
def home_positions(shots_df, index):
    position_array_home = []
    position_array_home.append(shots_df.Home_p1_pos[index])
    position_array_home.append(shots_df.Home_p2_pos[index])
    position_array_home.append(shots_df.Home_p3_pos[index])
    position_array_home.append(shots_df.Home_p4_pos[index])
    position_array_home.append(shots_df.Home_p5_pos[index])
    position_array_home.append(shots_df.Home_p6_pos[index])
    return position_array_home


# Create array of away team positions on ice
def away_positions(shots_df, index):
    position_array_away = []
    position_array_away.append(shots_df.Away_p1_pos[index])
    position_array_away.append(shots_df.Away_p2_pos[index])
    position_array_away.append(shots_df.Away_p3_pos[index])
    position_array_away.append(shots_df.Away_p4_pos[index])
    position_array_away.append(shots_df.Away_p5_pos[index])
    position_array_away.append(shots_df.Away_p6_pos[index])
    return position_array_away


def home_numbers(shots_df, index):
    number_array_home = []
    number_array_home.append(shots_df.Home_p1_num[index])
    number_array_home.append(shots_df.Home_p2_num[index])
    number_array_home.append(shots_df.Home_p3_num[index])
    number_array_home.append(shots_df.Home_p4_num[index])
    number_array_home.append(shots_df.Home_p5_num[index])
    number_array_home.append(shots_df.Home_p6_num[index])
    return number_array_home


# Create array of away team positions on ice
def away_numbers(shots_df, index):
    number_array_away = []
    number_array_away.append(shots_df.Away_p1_num[index])
    number_array_away.append(shots_df.Away_p2_num[index])
    number_array_away.append(shots_df.Away_p3_num[index])
    number_array_away.append(shots_df.Away_p4_num[index])
    number_array_away.append(shots_df.Away_p5_num[index])
    number_array_away.append(shots_df.Away_p6_num[index])
    return number_array_away


# Create arrays for each home and away team (which players are on ice for each time)
home_pos_array = []
away_pos_array = []
home_num_array = []
away_num_array = []
for i in shots.index.values:
    home_pos_array.append(home_positions(shots, i))
    away_pos_array.append(away_positions(shots, i))
    home_num_array.append(home_numbers(shots, i))
    away_num_array.append(away_numbers(shots, i))

# Add arrays to df
shots['home_pos_array'] = home_pos_array
shots['away_pos_array'] = away_pos_array
shots['home_num_array'] = home_num_array
shots['away_num_array'] = away_num_array


############################################################################################
# Identify empty net goals
############################################################################################

empty_net_goal = []
for i in shots.index.values:

    # If there was a goal score
    if (shots['Description'][i] == 'GOAL'):

        # Scoring team
        scoring_team = shots['team_for'][i]

        # If scoring team is home, check that opposing team had a goalie in net
        if ((scoring_team == shots['home_team_abb'][i]) & ('Goalie' not in shots['away_pos_array'][i])):
            empty_net_goal.append(1)
        elif ((scoring_team == shots['away_team_abb'][i]) & ('Goalie' not in shots['home_pos_array'][i])):
            empty_net_goal.append(1)
        else:
            empty_net_goal.append(0)

    # If no goal was scored
    else:
        empty_net_goal.append(0)

shots['empty_net_goal'] = empty_net_goal

print('Empty net goals removed')

############################################################################################
# Calculate shot angles
############################################################################################

def calc_angle_from_middle(x, y):
    opp = abs(y)
    adj = 100-abs(x)
    radian = math.atan(opp/adj)
    degree = radian * 180/math.pi
    return degree

shots['angle'] = list(map(lambda x,y: calc_angle_from_middle(x, y),
                                shots['x_coord'], shots['y_coord']))

shots['shot_dist'] = list(map(lambda x: int(x.split(',')[len(x.split(','))-1][1:].split(' ')[0]), shots['Details']))
shots['shot_type'] = list(map(lambda x: x.split(',')[1], shots['Details']))

print('Shot angles added')

############################################################################################
# Add final training set info
############################################################################################

# Game situation
conditions = [(shots['Strength'] =='SH'),
             (shots['Strength'] =='EV'),
             (shots['Strength'] =='PP')]
tranche = [0,1,2]
shots['strength_num'] = np.select(conditions,tranche)

# Shot type
shots['shot_wrist'] = np.where(shots['shot_type']==' Wrist',1,0)
shots['shot_slap'] = np.where(shots['shot_type']==' Slap',1,0)
shots['shot_snap'] = np.where(shots['shot_type']==' Snap',1,0)
shots['shot_backhand'] = np.where(shots['shot_type']==' Backhand',1,0)

# Goal
shots['goal_bin'] = np.where(shots['Description']=='GOAL',1,0)


############################################################################################
# Modeling
############################################################################################

# Create balanced dataset (half goals, half not)
goals = shots[shots['goal_bin'] == 1]
non_goals = shots[shots['goal_bin'] == 0].sample(n=goals.shape[0])
short_df = shuffle(goals.append(non_goals))

# Create x and y sets
df_y_short = short_df['goal_bin']
df_x_short = short_df[['angle', 'shot_dist','strength_num','shot_wrist',
                 'shot_slap','shot_backhand']].astype(int)

############################################################################################
# Finding best model hyperparameters
############################################################################################

# Creating the random grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 150, num = 11)]
#### max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 5]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Training model
rf = RandomForestClassifier(random_state=42)

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    #n_iter=20,
    # scoring='recall', # if using classification
    cv=2,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
)

rf_random.fit(df_x_short, df_y_short)

#model = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=10,
#                                n_estimators=118, random_state=42).fit(df_x_short, df_y_short)
model = rf_random.best_estimator_
y_pred = model.predict(df_x_short)

# Assessing model importance
#feature_importance = model.feature_importances_
#sorted_idx = np.argsort(feature_importance)
#pos = np.arange(sorted_idx.shape[0]) + .5
#fig = plt.figure(figsize=(24, 12))
#plt.subplot(1, 2, 1)
#plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, np.array(df_x_short.columns)[sorted_idx])
#plt.title('Feature Importance (Random Forest)')

# Evaluating model
c_matrix_test = confusion_matrix(df_y_short, y_pred)
acc = np.round(sklearn.metrics.accuracy_score(df_y_short, y_pred), 5)
prec = np.round(sklearn.metrics.precision_score(df_y_short, y_pred, average=None), 3)
prec_micro = np.round(sklearn.metrics.precision_score(df_y_short, y_pred, average='micro'), 5)
rec = np.round(sklearn.metrics.recall_score(df_y_short, y_pred, average=None), 3)
rec_micro = np.round(sklearn.metrics.recall_score(df_y_short, y_pred, average='micro'), 5)
f1 = np.round(sklearn.metrics.f1_score(df_y_short, y_pred, average=None), 3)
f1_micro = np.round(sklearn.metrics.f1_score(df_y_short, y_pred, average='micro'), 5)
print('Acc: ', acc, ' Prec: ', prec, ' Rec: ', rec, ' f1: ', f1)

# Plot confusion matrix
#cmd_obj = ConfusionMatrixDisplay(c_matrix_test, display_labels=['no goal', 'goal'])
#cmd_obj.plot()
#cmd_obj.ax_.set(
               # title='Confusion Matrix',
               # xlabel='Predicted behavior',
               # ylabel='Actual behavior')
#plt.show()

# Partial dependance of variables
#for i in range(0, len(df_x_short.columns)):
#    plot_partial_dependence(model, df_x_short, [i])
#    plt.show()
#    print('   ---   ---   ---   ---   ---   ---   ')

print('Model trained using random grid search cv')

############################################################################################
# Predict goal probability on full set
############################################################################################

shots_predictors = shots[['angle', 'shot_dist','strength_num','shot_wrist',
                 'shot_slap','shot_backhand']].astype(int)

predictions = model.predict_proba(shots_predictors)
shots['goal_prob'] = list(map(lambda x: x[1], predictions))

# save the model to disk
filename = 'data/model/test_model_dump.sav'
pickle.dump(model, open(filename, 'wb'))

# Read in model
loaded_model = pickle.load(open('data/model/test_model_dump.sav', 'rb'))

# Make predictions
pred_loaded_model = loaded_model.predict_proba(shots_predictors)
shots['goal_prob_loaded_model'] = list(map(lambda x: x[1], pred_loaded_model))


# Round to nearest 0.05
def round_nearest(num: float) -> float:
    num, to = Decimal(str(num)), Decimal(0.05)
    return float(round(num / to) * to)

shots['rounded_prob'] = list(map(lambda x: round(round_nearest(x),4), shots['goal_prob']))


#####################################################################################
### Convert probabilities to expected goals
#####################################################################################

# Observed goals scored per probability tranche
summary = pd.pivot_table(shots, values='goal_bin', index=['rounded_prob'], aggfunc=['count', np.sum])

prob_goal_mapping = pd.DataFrame({'rounded_prob':(summary['sum']['goal_bin'] / summary['count']['goal_bin']).index,
                                  'xG':(summary['sum']['goal_bin'] / summary['count']['goal_bin']).values})
print('Goal probability to xG mapping:')
print(prob_goal_mapping)

print('Expected goals merged to shots df')
shots = shots.merge(prob_goal_mapping, how='left', on='rounded_prob')

#####################################################################################
### Save shots
#####################################################################################

shots.to_csv('data/shots_and_xG.csv')
print('Shots df saved')


#####################################################################################
### Summarize by player - one row per player per game
#####################################################################################

game_summary = pd.read_csv('data/game_summary_2021.csv')
game_summary = game_summary.reset_index()
game_summary['game'] = list(map(lambda x: int(x.split('-')[1]), game_summary.year_game_home_away))
game_summary['year'] = list(map(lambda x: int(x.split('-')[0][0:4]), game_summary.year_game_home_away))

# Adjust team names for accents
team_dict = pd.DataFrame({'team_name': ['ANAHEIM DUCKS',
  'ARIZONA COYOTES',
  'BOSTON BRUINS',
  'BUFFALO SABRES',
  'CAROLINA HURRICANES',
  'COLUMBUS BLUE JACKETS',
  'CALGARY FLAMES',
  'CHICAGO BLACKHAWKS',
  'COLORADO AVALANCHE',
  'DALLAS STARS',
  'DETROIT RED WINGS',
  'EDMONTON OILERS',
  'FLORIDA PANTHERS',
  'LOS ANGELES KINGS',
  'MINNESOTA WILD',
  'MONTREAL CANADIENS',
  'MONTRÉAL CANADIENS',
  'NEW JERSEY DEVILS',
  'NASHVILLE PREDATORS',
  'NEW YORK ISLANDERS',
  'NEW YORK RANGERS',
  'OTTAWA SENATORS',
  'PHILADELPHIA FLYERS',
  'PITTSBURGH PENGUINS',
  'SAN JOSE SHARKS',
  'SEATTLE KRAKEN',
  'ST. LOUIS BLUES',
  'TAMPA BAY LIGHTNING',
  'TORONTO MAPLE LEAFS',
  'VEGAS GOLDEN KNIGHTS',
  'VANCOUVER CANUCKS',
  'WINNIPEG JETS',
  'WASHINGTON CAPITALS'],
 'team_abb': ['ANA',
  'ARI',
  'BOS',
  'BUF',
  'CAR',
  'CBJ',
  'CGY',
  'CHI',
  'COL',
  'DAL',
  'DET',
  'EDM',
  'FLA',
  'LAK',
  'MIN',
  'MTL',
  'MTL',
  'NJD',
  'NSH',
  'NYI',
  'NYR',
  'OTT',
  'PHI',
  'PIT',
  'SJS',
  'SEA',
  'STL',
  'TBL',
  'TOR',
  'VGK',
  'VAN',
  'WPG',
  'WSH']})
game_summary = game_summary.merge(team_dict, how = 'left', left_on='team', right_on='team_name')

# Finding the most recent team for each player
game_summary=game_summary[game_summary['year']>=2021]
game_summary= game_summary.sort_values(by='game')
player_most_recent_team = game_summary[['player_name','team_abb']].groupby('player_name').tail(1)

# Identify unique games
unique_games = game_summary.year_game_home_away.value_counts().index


def game_summary_with_XG(shots, game_summary, game_id):
    # Subset game summary for game only
    subset_summary = game_summary[game_summary['year_game_home_away'] == game_id]

    # Subset shots for game only
    subset_shots = shots[shots['year_game_home_away'] == game_id]
    subset_shots['team_for_players'] = np.where(subset_shots.team_for == subset_shots.home_team_abb,
                                                subset_shots.home_num_array, subset_shots.away_num_array)
    subset_shots['team_against_players'] = np.where(subset_shots.team_for == subset_shots.away_team_abb,
                                                    subset_shots.home_num_array, subset_shots.away_num_array)

    # Create summary
    EV_for_agg = []
    PP_for_agg = []
    PK_for_agg = []
    EV_against_agg = []
    PP_against_agg = []
    PK_against_agg = []

    for i in range(0, subset_summary.shape[0]):

        # For each individual player
        player_team = subset_summary.team_abb.iloc[i]
        player_num = subset_summary.player_number.iloc[i]

        EV_for = []
        PP_for = []
        PK_for = []
        EV_against = []
        PP_against = []
        PK_against = []

        # For each row of gameplay, check if player was involved and compile stats
        for j in range(0, subset_shots.shape[0]):

            # If player is on ice as "for team"
            if ((player_team == subset_shots.team_for.iloc[j]) & \
                    (str(player_num) in subset_shots.team_for_players.iloc[j])):
                # print(subset_shots.team_for_players.iloc[j])
                if subset_shots.Strength.iloc[j] == 'EV':
                    EV_for.append(subset_shots.xG.iloc[j])
                elif subset_shots.Strength.iloc[j] == 'PP':
                    PP_for.append(subset_shots.xG.iloc[j])
                else:
                    PK_for.append(subset_shots.xG.iloc[j])

            # If player is on ice as "Against team"
            if (str(player_num) in subset_shots.team_against_players.iloc[j]) & \
                    (player_team != subset_shots.team_for.iloc[j]):
                # print(subset_shots.team_against_players.iloc[j])
                if subset_shots.Strength.iloc[j] == 'EV':
                    EV_against.append(subset_shots.xG.iloc[j])
                elif subset_shots.Strength.iloc[j] == 'PP':
                    PK_against.append(subset_shots.xG.iloc[j])
                else:
                    PP_against.append(subset_shots.xG.iloc[j])

        # Add up stats for player
        EV_for_agg.append(sum(EV_for))
        PP_for_agg.append(sum(PP_for))
        PK_for_agg.append(sum(PK_for))
        EV_against_agg.append(sum(EV_against))
        PP_against_agg.append(sum(PP_against))
        PK_against_agg.append(sum(PK_against))

    subset_summary['EV_for_agg'] = EV_for_agg
    subset_summary['PP_for_agg'] = PP_for_agg
    subset_summary['PK_for_agg'] = PK_for_agg
    subset_summary['EV_against_agg'] = EV_against_agg
    subset_summary['PP_against_agg'] = PP_against_agg
    subset_summary['PK_against_agg'] = PK_against_agg

    return subset_summary

# Run function
games_computed = []
games_failed = []

# Create empty pd
game_summaries_with_xG_agg = pd.DataFrame()

for game in unique_games:
    try:
        unique_game_summary = game_summary_with_XG(shots, game_summary, game)
        game_summaries_with_xG_agg = game_summaries_with_xG_agg.append(unique_game_summary)
        games_computed.append(game)
    except:
        games_failed.append(game)

print('xG stats aggregated by player for each game')
print('Games successful:')
print(games_computed)
print('Games failed:')
print(games_failed)

print('data_saved')
game_summaries_with_xG_agg.to_csv('data/game_sums_with_XG_21.csv')

#####################################################################################
### Aggregate per player (one row per player
#####################################################################################

# Removing goaltenders (for who we dont have playing minutes)
game_summaries_with_xG_agg =game_summaries_with_xG_agg[game_summaries_with_xG_agg['position']!='G']

# Finding year
game_summaries_with_xG_agg['year'] = list(map(lambda x: x[0:4], game_summaries_with_xG_agg.year_game_home_away))

# Remove players with no TOI at all (there are just 4 in 3 years)
game_summaries_with_xG_agg['len_toi'] = list(map(lambda x: len(x), game_summaries_with_xG_agg.TOI))
game_summaries_with_xG_agg = game_summaries_with_xG_agg[game_summaries_with_xG_agg['len_toi'] == 5]


def convert_string_to_seconds(string_time):
    'Converts string time (min:sec) to seconds'

    # Deal with minutes
    minutes = list(map(lambda x: x.split(':')[0], np.array(string_time)))
    minutes = np.where(np.array(minutes) == '\xa0', '0', np.array(minutes))
    minutes = minutes.astype('int')

    # Deal with seconds
    seconds = list(map(lambda x: x.split(':')[1], np.array(string_time)))
    seconds = np.where(np.array(seconds) == '\xa0', '0', np.array(seconds))
    seconds = seconds.astype('int')

    # Total seconds
    total_seconds = 60 * minutes + seconds
    return total_seconds

# Converting time to seconds
game_summaries_with_xG_agg['TOI_seconds'] = convert_string_to_seconds(game_summaries_with_xG_agg.TOI)
game_summaries_with_xG_agg['EV_TOI_seconds'] = convert_string_to_seconds(game_summaries_with_xG_agg.EV_TOI)
game_summaries_with_xG_agg['PP_TOI_seconds'] = convert_string_to_seconds(game_summaries_with_xG_agg.PP_TOI)
game_summaries_with_xG_agg['SH_TOI_seconds'] = convert_string_to_seconds(game_summaries_with_xG_agg.SH_TOI)

# Adjust for empty space in game sheets
game_summaries_with_xG_agg['Goals'] = np.where(game_summaries_with_xG_agg['Goals']== ' ', 0,
                                               game_summaries_with_xG_agg['Goals'])
game_summaries_with_xG_agg['Assists'] = np.where(game_summaries_with_xG_agg['Assists']== ' ', 0,
                                               game_summaries_with_xG_agg['Assists'])
game_summaries_with_xG_agg['PIM'] = np.where(game_summaries_with_xG_agg['PIM']== ' ', 0,
                                               game_summaries_with_xG_agg['PIM'])
game_summaries_with_xG_agg['Shots'] = np.where(game_summaries_with_xG_agg['Shots']== ' ', 0,
                                               game_summaries_with_xG_agg['Shots'])
game_summaries_with_xG_agg['Hits'] = np.where(game_summaries_with_xG_agg['Hits']== ' ', 0,
                                               game_summaries_with_xG_agg['Hits'])

# Adjust for '\xa0' in game sheets
game_summaries_with_xG_agg['Goals'] = np.where(game_summaries_with_xG_agg['Goals']== '\xa0', 0,
                                               game_summaries_with_xG_agg['Goals'])
game_summaries_with_xG_agg['Assists'] = np.where(game_summaries_with_xG_agg['Assists']== '\xa0', 0,
                                               game_summaries_with_xG_agg['Assists'])
game_summaries_with_xG_agg['PIM'] = np.where(game_summaries_with_xG_agg['PIM']== '\xa0', 0,
                                               game_summaries_with_xG_agg['PIM'])
game_summaries_with_xG_agg['Shots'] = np.where(game_summaries_with_xG_agg['Shots']== '\xa0', 0,
                                               game_summaries_with_xG_agg['Shots'])
game_summaries_with_xG_agg['Hits'] = np.where(game_summaries_with_xG_agg['Hits']== '\xa0', 0,
                                               game_summaries_with_xG_agg['Hits'])

# Convert column types
game_summaries_with_xG_agg[['TOI_seconds', 'EV_TOI_seconds', 'PP_TOI_seconds', 'SH_TOI_seconds',
                                     'EV_for_agg', 'PP_for_agg', 'PK_for_agg',
                                      'EV_against_agg', 'PP_against_agg', 'PK_against_agg',
                                     'Goals', 'Assists', 'PIM', 'Shots', 'Hits']] = \
                                    game_summaries_with_xG_agg[['TOI_seconds', 'EV_TOI_seconds', 'PP_TOI_seconds',
                                                               'SH_TOI_seconds', 'EV_for_agg', 'PP_for_agg',
                                                               'PK_for_agg', 'EV_against_agg', 'PP_against_agg',
                                                               'PK_against_agg', 'Goals', 'Assists',
                                                               'PIM',
                                                                'Shots',
                                                                'Hits']].apply(pd.to_numeric)

# Retain only 2021 data
agg_xg_2021 = game_summaries_with_xG_agg[game_summaries_with_xG_agg['year']=='2021']

# Extract players most recent position
player_most_recent_position = game_summaries_with_xG_agg[['player_name','player_number','position']].groupby('player_name').tail(1)

# Pivot to aggregate by player
pivot_v1 = pd.pivot_table(agg_xg_2021,
                            values = ['TOI_seconds', 'EV_TOI_seconds', 'PP_TOI_seconds', 'SH_TOI_seconds',
                                     'EV_for_agg', 'PP_for_agg', 'PK_for_agg',
                                      'EV_against_agg', 'PP_against_agg', 'PK_against_agg',
                                     'Goals', 'Assists', 'Rating', 'PIM', 'Shots', 'Hits'
                                     ],
                            index = 'player_name',
                            aggfunc=['sum','count'])

# Clean
pivot_v2 = pd.DataFrame({'Player': pivot_v1.index.values,
              'GP': pivot_v1['count']['EV_TOI_seconds'].values,
              'EV_TOI_seconds': pivot_v1['sum']['EV_TOI_seconds'].values,
              'PP_TOI_seconds': pivot_v1['sum']['PP_TOI_seconds'].values,
              'SH_TOI_seconds': pivot_v1['sum']['SH_TOI_seconds'].values,
              'EV_for_agg': pivot_v1['sum']['EV_for_agg'].values,
              'PP_for_agg': pivot_v1['sum']['PP_for_agg'].values,
              'PK_for_agg': pivot_v1['sum']['PK_for_agg'].values,
              'EV_against_agg': pivot_v1['sum']['EV_against_agg'].values,
              'PP_against_agg': pivot_v1['sum']['PP_against_agg'].values,
              'PK_against_agg': pivot_v1['sum']['PK_against_agg'].values,
              'Shots': pivot_v1['sum']['Shots'].values,
              'Goals': pivot_v1['sum']['Goals'].values,
              'Assists': pivot_v1['sum']['Assists'].values,
              'Hits': pivot_v1['sum']['Hits'].values,
              'PIM': pivot_v1['sum']['PIM'].values
              })

# Add calculated fields
pivot_v2['EV_for_per_60'] = pivot_v2['EV_for_agg'] / (pivot_v2['EV_TOI_seconds']/3600)
pivot_v2['PP_for_per_60'] = pivot_v2['PP_for_agg'] / (pivot_v2['PP_TOI_seconds']/3600)
pivot_v2['PK_for_per_60'] = pivot_v2['PK_for_agg'] / (pivot_v2['SH_TOI_seconds']/3600)
pivot_v2['EV_against_per_60'] = pivot_v2['EV_against_agg'] / (pivot_v2['EV_TOI_seconds']/3600)
pivot_v2['PP_against_per_60'] = pivot_v2['PP_against_agg'] / (pivot_v2['PP_TOI_seconds']/3600)
pivot_v2['PK_against_per_60'] = pivot_v2['PK_against_agg'] / (pivot_v2['SH_TOI_seconds']/3600)

pivot_v2['EV_netXG'] = pivot_v2['EV_for_agg'] - pivot_v2['EV_against_agg']
pivot_v2['EV_netXG_per_60'] = pivot_v2['EV_netXG'] / (pivot_v2['EV_TOI_seconds']/3600)
pivot_v2['clutch_factor'] = (pivot_v2['Assists'] + pivot_v2['Goals'])/ \
                            (pivot_v2['EV_for_agg'] + pivot_v2['PP_for_agg'] + pivot_v2['PK_for_agg'])

# Sort from highest to lowest net xG
pivot_v2 = pivot_v2.sort_values(by='EV_netXG_per_60', ascending=False)

# Add most recent team and position for each player
pivot_v2 = pivot_v2.merge(player_most_recent_team, how='left', left_on='Player', right_on='player_name')
pivot_v2 = pivot_v2.merge(player_most_recent_position, how='left', left_on='Player', right_on='player_name')

# Read in salaries
salaries_data = pd.read_csv('data/capfriendly_salaries_2021.csv')
salaries_data['clean_name'] = list(map(lambda x: str(str(x.split('-')[1])+', '+str(x.split('-')[0])),
                                       salaries_data.player_name))
# Filtering for 2021 season
salaries_data_min=salaries_data[salaries_data['season']=='2021-22'][['clean_name','cap_hit']]
salaries_data_min=salaries_data_min.fillna(0)
salaries_data_min=salaries_data_min[salaries_data_min['cap_hit']!=0]
# Ensuring that I only get one salary per player
salaries_data_min = salaries_data_min[['clean_name','cap_hit']].groupby('clean_name').head(1)

# Add salary for each player
pivot_v2 = pivot_v2.merge(salaries_data_min, how='left', left_on='Player', right_on='clean_name')

# Convert salary strings to numeric
pivot_v3 = pivot_v2.fillna('$0')
pivot_v3['num_salary']= list(map(lambda x: int(x.split('$')[1].replace(',','')), pivot_v3.cap_hit))

# Save data
pivot_v3.to_csv('data/player_summary_with_xG_salary_2021.csv')

print('Top player sample:')
print(pivot_v3[pivot_v3['GP']>40][0:50])

# Done!
