import random
import time

import pandas as pd

from src.play_by_play import get_game_play
from src.game_summary import game_summary_to_pd_2
from src.x_y_coord import game_details_coord

# Declare which games to lookup
total_games = int(82 * 32 / 2)
# games = list(range(1, total_games))
games = [1, 2, 13, 20, 25, 39, 30, 833, 810, 526, 529, 858, 860]

#################################################
# Play by play
#################################################

full_df = pd.DataFrame()
games_scraped = []
failed_scrape = []

# Extract game plays
for game_num in games:

    # Random sleeper
    random_number = random.randint(1, 200) / 100
    time.sleep(random_number)

    # Extract data for current game
    try:
        # current_game = get_game_play(2019, game_num)
        # current_game = get_game_play(2020, game_num)
        current_game = get_game_play(2021, game_num)
        full_df = full_df.append(current_game)
        games_scraped.append(game_num)
    except:
        failed_scrape.append(game_num)

full_df

# Clean df (for team names)
# Create dict of team names and abbreviations
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

# Identify home and away team
full_df['home_full'] = list(map(lambda x: x.split('-')[2], full_df.year_game_home_away))
full_df['away_full'] = list(map(lambda x: x.split('-')[3], full_df.year_game_home_away))

# Identify the abbreviation of the home team
full_df = full_df.merge(team_dict, how='left', left_on='home_full', right_on='team_name')
full_df.rename(columns={'team_abb': 'home_team_abb'}, inplace=True)
full_df = full_df.drop(columns=['team_name'])

# Identify the abbreviation of the away team
full_df = full_df.merge(team_dict, how='left', left_on='away_full', right_on='team_name')
full_df.rename(columns={'team_abb': 'away_team_abb'}, inplace=True)
full_df = full_df.drop(columns=['team_name'])

# Save data
# full_df.to_csv("data/play_by_play_2021.csv")

print('Play by play data downloaded')
# print(full_df.away_team_abb.value_counts())
print(full_df.shape)

print('successful scrapes:')
print(games_scraped)
print('failed scrapes:')
print(failed_scrape)


#################################################
# Game summary
#################################################

full_df = pd.DataFrame()
games_scraped = []
failed_scrape = []

for game_num in games:

    # Random sleeper
    random_number = random.randint(1, 200) / 100
    time.sleep(random_number)

    # Extract data for current game
    try:
        current_game = game_summary_to_pd_2(2021, game_num)
        full_df = full_df.append(current_game)
        games_scraped.append(game_num)
    except:
        failed_scrape.append(game_num)

# Replace unicode character 'xa)'
full_df['Shots'] = full_df['Shots'].apply(lambda x: str(x).replace(u'\xa0', u'0'))

# Convert columns to numeric
full_df["Shots"] = pd.to_numeric(full_df["Shots"])

# Save data
# full_df.to_csv("data/game_summary_2021.csv")

# Print messages
print('Game summary data downloaded')
# print(full_df.team.value_counts())
print(full_df.shape)

print('successful scrapes:')
print(games_scraped)
print('failed scrapes:')
print(failed_scrape)


#################################################
# X & Y coordinates
#################################################

total_games = int(82 * 32 / 2)
games = list(range(1, total_games))
# games = [1, 2, 13, 20, 25, 39, 30, 833, 810, 526, 529, 858, 860]

full_df = pd.DataFrame()
games_scraped = []
failed_scrape = []

for game_num in games:

    # Random sleeper
    random_number = random.randint(1, 200) / 100
    time.sleep(random_number)

    # Extract data for current game
    try:
        current_game = game_details_coord(2021, game_num)
        full_df = full_df.append(current_game)
        games_scraped.append(game_num)
    except:
        failed_scrape.append(game_num)

print('X & Y coordinates data downloaded')
print(full_df.shape)

print('successful scrapes:')
print(games_scraped)
print('failed scrapes:')
print(failed_scrape)

full_df.to_csv("play_by_play_coord_2021.csv")