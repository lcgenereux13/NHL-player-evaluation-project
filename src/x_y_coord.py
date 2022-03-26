import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import regex
import requests
import time
import warnings
from bs4 import BeautifulSoup
from matplotlib.pyplot import figure

warnings.filterwarnings("ignore")


def game_details_coord(year, game_number):
    # Create url for game
    season_strings = str(year)

    game_number_string = str(game_number)
    if len(game_number_string) == 1:
        game_number_string = str(str('000') + str(game_number))
    elif len(game_number_string) == 2:
        game_number_string = str(str('00') + str(game_number))
    elif len(game_number_string) == 3:
        game_number_string = str(str('0') + str(game_number))

    full_web_address = "https://statsapi.web.nhl.com/api/v1/game/" + season_strings + "02" + game_number_string + "/feed/live?site=en_nhl"

    # Extract data from web page
    page_play_by_play = requests.get(full_web_address)
    soup_play_by_play = BeautifulSoup(page_play_by_play.content, 'html.parser')

    # Convert to json format
    json_play_by_play = json.loads(soup_play_by_play.text)

    # Create pd to store data
    pd_data = pd.DataFrame(json_play_by_play['liveData']['plays']['allPlays'])

    # Remove events where period ends
    pd_data = pd_data[pd_data['players'].notna()].reset_index()

    # Remove lines where there is no x-y location
    pd_data['coord_len'] = list(map(lambda x: len(x),
                                    pd_data['coordinates']))
    pd_data = pd_data[pd_data['coordinates'].notna()].reset_index()
    pd_data = pd_data[pd_data['coord_len'] == 2]

    # Create new columns
    pd_data['event'] = list(map(lambda x: x['event'],
                                pd_data['result']))
    pd_data['description'] = list(map(lambda x: x['description'],
                                      pd_data['result']))
    pd_data['period'] = list(map(lambda x: x['period'],
                                 pd_data['about']))
    pd_data['periodTime'] = list(map(lambda x: x['periodTime'],
                                     pd_data['about']))
    pd_data['periodTimeRemaining'] = list(map(lambda x: x['periodTimeRemaining'],
                                              pd_data['about']))
    pd_data['x_coord'] = list(map(lambda x: x['x'],
                                  pd_data['coordinates']))
    pd_data['y_coord'] = list(map(lambda x: x['y'],
                                  pd_data['coordinates']))

    # Return data
    pd_data_final = pd_data[['period', 'periodTime', 'periodTimeRemaining', 'event', 'description',
                             'x_coord', 'y_coord']]

    # Add new column specific to game
    season_strings = str(year) + str(year + 1)
    home_team_name = json_play_by_play['gameData']['teams']['home']['name'].upper()
    away_team_name = json_play_by_play['gameData']['teams']['away']['name'].upper()

    pd_data_final['year_game_home_away'] = str(
        str(season_strings) + '-' + str(game_number) + '-' + home_team_name + '-' + away_team_name)

    return pd_data_final


