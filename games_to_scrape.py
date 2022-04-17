######
# To be deleted
######

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
#games = [1, 2, 13, 20, 25, 39, 30, 833, 810, 526, 529, 858, 860]

# list(set(list(play_by_play['clean_game'])) - set(list(x_y_coord['clean_game'])))

existing_df = pd.read_csv("data/capfriendly_salaries_2021.csv")
print(existing_df.columns)
full_df = pd.DataFrame()
existing_df = existing_df.append(full_df)
print(existing_df.head())
print(existing_df.columns)

play_by_play_2021 = pd.read_csv('data/play_by_play_2021.csv')

names = []
names_col = ['Away_p1_name', 'Away_p2_name', 'Away_p3_name',
             'Away_p4_name', 'Away_p5_name', 'Away_p6_name',
             'Home_p1_name', 'Home_p2_name', 'Home_p3_name',
             'Home_p4_name', 'Home_p5_name', 'Home_p6_name']

for col in names_col:
    names.extend(play_by_play_2021[col].tolist())

player_list = list(pd.Series(names).value_counts().index)

print(player_list[0:10])


# scraped_already = list(existing_df.player_name.value_counts().index)
scraped_already = list(map(lambda x: str(x.split('-')[0] + ' ' + x.split('-')[1]), list(existing_df.player_name.value_counts().index)))
print(scraped_already[0:10])
#left_to_scrape = list(set(list(existing_df.player_name)) - set(list(scraped_already)))
#print((left_to_scrape))
# existing_df.to_csv("data/play_by_play_2021.csv")
