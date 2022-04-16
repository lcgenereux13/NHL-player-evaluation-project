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
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(24, 12))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(df_x_short.columns)[sorted_idx])
plt.title('Feature Importance (Random Forest)')

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
cmd_obj = ConfusionMatrixDisplay(c_matrix_test, display_labels=['no goal', 'goal'])
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Confusion Matrix',
                xlabel='Predicted behavior',
                ylabel='Actual behavior')
plt.show()

# Partial dependance of variables
#for i in range(0, len(df_x_short.columns)):
#    plot_partial_dependence(model, df_x_short, [i])
#    plt.show()
#    print('   ---   ---   ---   ---   ---   ---   ')


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