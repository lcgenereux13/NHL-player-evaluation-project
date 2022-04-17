import pandas as pd
import random
import time
from src.capfriendly_scrape import get_salary_info
from src.game_summary import game_summary_to_pd_2
from src.play_by_play import get_game_play
from src.x_y_coord import game_details_coord

# Declare which games to lookup
total_games = int(82 * 32 / 2)
games = list(range(1, total_games))
# games = [1, 2, 13, 20, 25, 39, 30, 833, 810, 526, 529, 858, 860]

#################################################
# Play by play
#################################################

# Read in existing data, determine which gaimes are remaining
existing_df = pd.read_csv("data/play_by_play_2021.csv")
scraped_already = list(map(lambda x: int(x.split('-')[1]), existing_df.year_game_home_away))
left_to_scrape = list(set(list(games)) - set(list(scraped_already)))

full_df = pd.DataFrame()
games_scraped = []
failed_scrape = []

# Extract game plays
for game_num in left_to_scrape:

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
existing_df = existing_df.append(full_df)
existing_df.to_csv("data/play_by_play_2021.csv")

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

# Read in existing data, determine which games are remaining
existing_df = pd.read_csv("data/game_summary_2021.csv")
scraped_already = list(map(lambda x: int(x.split('-')[1]), existing_df.year_game_home_away))
left_to_scrape = list(set(list(games)) - set(list(scraped_already)))

full_df = pd.DataFrame()
games_scraped = []
failed_scrape = []

for game_num in left_to_scrape:

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
existing_df = existing_df.append(full_df)
existing_df.to_csv("data/game_summary_2021.csv")

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

# Read in existing data, determine which gaimes are remaining
existing_df = pd.read_csv("data/play_by_play_coord_2021.csv")
scraped_already = list(map(lambda x: int(x.split('-')[1]), existing_df.year_game_home_away))
left_to_scrape = list(set(list(games)) - set(list(scraped_already)))


full_df = pd.DataFrame()
games_scraped = []
failed_scrape = []

for game_num in left_to_scrape:

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

# Save data
existing_df = existing_df.append(full_df)
existing_df.to_csv("data/play_by_play_coord_2021.csv")


#################################################
# Capfriendly salaries
#################################################

# Extract all the names for which salaries should be scraped
play_by_play_2021 = pd.read_csv('data/play_by_play_2021.csv')

names = []
names_col = ['Away_p1_name', 'Away_p2_name', 'Away_p3_name',
             'Away_p4_name', 'Away_p5_name', 'Away_p6_name',
             'Home_p1_name', 'Home_p2_name', 'Home_p3_name',
             'Home_p4_name', 'Home_p5_name', 'Home_p6_name']

for col in names_col:
    names.extend(play_by_play_2021[col].tolist())

player_list = pd.Series(names).value_counts().index

# Find only new names that need to be scraped
existing_df = pd.read_csv("data/capfriendly_salaries_2021.csv")
# scraped_already = existing_df.player_name.value_counts().index
scraped_already = list(map(lambda x: str(x.split('-')[0] + ' ' + x.split('-')[1]),
                           list(existing_df.player_name.value_counts().index)))


# Remove unsupported characters
def clean_player_name(player_name):
    player_name = player_name.replace('.', '')
    player_name = player_name.replace("'", '')
    player_name = player_name.replace('.', '')
    player_name = player_name.replace('Ü', 'U')
    player_name = player_name.replace('È', 'E')
    player_name = player_name.replace('É', 'E')
    return player_name


scraped_already = list(map(lambda x: clean_player_name(x), scraped_already))
player_list = list(map(lambda x: clean_player_name(x), player_list))

left_to_scrape = list(set(list(player_list)) - set(list(scraped_already)))

# Go through each player
full_df = pd.DataFrame()
players_scraped = []
failed_scrape = []

for player in left_to_scrape:

    first = player.split(' ')[0]
    last = player.split(' ')[1]

    # Random sleeper
    random_number = random.randint(1, 200) / 100
    time.sleep(random_number)

    # Extract data for current player
    try:
        current_player = get_salary_info(str(first), str(last))
        full_df = full_df.append(current_player)
        players_scraped.append(player)
    except:
        failed_scrape.append(player)

print('New scrapes:')
print(full_df)

# Save data
existing_df = existing_df.append(full_df)
existing_df.to_csv("data/capfriendly_salaries_2021.csv")

print('Salaries data downloaded')
print(full_df.shape)

print('successful scrapes:')
print(players_scraped)
print('failed scrapes:')
print(failed_scrape)