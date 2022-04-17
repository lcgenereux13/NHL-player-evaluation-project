######
# To be deleted
######


from src.play_by_play import get_game_play
import pandas as pd



# Ad hoc add of missing game

game_play = pd.read_csv('data/play_by_play_2021.csv')
print(game_play.shape)
full_df = get_game_play(2021, 898)
print(full_df)
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

# Adding new game
game_play = game_play.append(full_df)
print(game_play.shape)

# Save as csv
# game_play.to_csv('data/play_by_play_2021.csv')
print('new game saved')

print('Game summaries with Xg')
print(pd.read_csv('data/game_sums_with_XG_21.csv').shape)
gam_sum_xg = pd.read_csv('data/game_sums_with_XG_21.csv')
print(len(gam_sum_xg.year_game_home_away.value_counts()))