import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd


def plot_game_summary(season, game_number):
    ''' Takes a game number and season, then plots the game summary'''

    # Read in data
    shots = pd.read_csv('data/shots_and_xG.csv')

    # Subset only data for the desired game
    subset_shots = shots[(shots['year'] == season) & (shots['game'] == game_number)]
    # print(subset_shots)

    # Prepare data prior to graphing
    subset_shots['cumulative_minutes'] = np.array(subset_shots['Period'] - 1) * 20 + \
                                         np.array(list(map(lambda x: int(x.split(':')[0]), subset_shots['Start'])))

    # Attributing goals to each team
    scoring_team = subset_shots['team_for']
    home_team_bin = np.where(np.array(scoring_team) == subset_shots['home_team_abb'], 1, 0)
    away_team_bin = np.where(np.array(scoring_team) == subset_shots['home_team_abb'], 0, 1)
    subset_shots['home_xG'] = (subset_shots.xG * home_team_bin)
    subset_shots['away_XG'] = (subset_shots.xG * away_team_bin)
    subset_shots['home_actual'] = (subset_shots.goal_bin * home_team_bin)
    subset_shots['away_actual'] = (subset_shots.goal_bin * away_team_bin)

    # Compute series for goals per minute
    time_cum, home_xG_cum, away_XG_cum, home_actual_cum, away_actual_cum = [], [], [], [], []

    for i in range(0, int(np.max(subset_shots.cumulative_minutes))):

        # Rows that match timestamp
        select_rows = subset_shots[subset_shots['cumulative_minutes'] == i]

        time_cum.append(i)

        # If there is action during that minute
        if (select_rows.shape[0] > 0):
            home_xG_cum.append(select_rows.home_xG.sum())
            away_XG_cum.append(select_rows.away_XG.sum())
            home_actual_cum.append(select_rows.home_actual.sum())
            away_actual_cum.append(select_rows.away_actual.sum())

        # If there is no action during that minute
        else:
            home_xG_cum.append(0)
            away_XG_cum.append(0)
            home_actual_cum.append(0)
            away_actual_cum.append(0)

    # Create cumulative summaries
    home_xG_cum = np.array(home_xG_cum).cumsum()
    away_XG_cum = np.array(away_XG_cum).cumsum()
    home_actual_cum = np.array(home_actual_cum).cumsum()
    away_actual_cum = np.array(away_actual_cum).cumsum()

    # Plotting
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(24, 12))

    plt.plot(time_cum, away_actual_cum, color='blue', label=str(str(subset_shots.away_team_abb.iloc[0]) + ' Goals'))
    plt.plot(time_cum, away_XG_cum, color='blue', alpha=0.2, label=str(str(subset_shots.away_team_abb.iloc[0]) + ' xG'))
    plt.plot(time_cum, home_actual_cum, color='red', label=str(str(subset_shots.home_team_abb.iloc[0]) + ' Goals'))
    plt.plot(time_cum, home_xG_cum, color='red', alpha=0.2, label=str(str(subset_shots.home_team_abb.iloc[0]) + ' xG'))

    plt.suptitle('Game summary', fontsize=30)
    plt.title(str('HOME: ' + str(subset_shots.home_full.iloc[0]) + \
                  '\nAWAY: ' + str(subset_shots.away_full.iloc[0])), fontsize=20, loc='left')

    plt.legend(loc=2, prop={'size': 15})
    plt.show()
