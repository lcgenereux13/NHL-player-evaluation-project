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
from bs4 import BeautifulSoup
from matplotlib.pyplot import figure


def extract_game_summary(year, game_number):
    ''' Takes the season and game number, scrapes the NHL website and returns html summary'''

    season_strings = str(year) + str(year + 1)

    game_number_string = str(game_number)

    if len(game_number_string) == 1:
        game_number_string = str(str('000') + str(game_number))
    elif len(game_number_string) == 2:
        game_number_string = str(str('00') + str(game_number))
    elif len(game_number_string) == 3:
        game_number_string = str(str('0') + str(game_number))

    full_web_address = "http://www.nhl.com/scores/htmlreports/" + season_strings + "/ES02" + game_number_string + ".HTM"

    # Extract data
    page_summary = requests.get(full_web_address)
    soup_summary = BeautifulSoup(page_summary.content, 'html.parser')

    # Identify home and away team
    regex = re.compile(r'alt="(.*)" border="')
    home_team_name = regex.findall(str(soup_summary.find_all('td', {'align': 'center'})[18]))[0]
    away_team_name = regex.findall(str(soup_summary.find_all('td', {'align': 'center'})[3]))[0]

    return soup_summary, home_team_name, away_team_name


def game_summary_to_pd_2(year, game_number):
    # Extract data
    soup_summary, home_team_name, away_team_name = extract_game_summary(year, game_number)

    ###########################################################################
    # Create first table
    ###########################################################################

    # Raw data from table
    table_data = soup_summary.find_all('td', {'align': 'center', 'class': 'rborder + bborder'})

    # Extract only useful player info
    regex = re.compile(r'>(.*)</td>')

    # Create lists
    player_name_list = []
    TOI_list = []
    PP_TOI_list = []
    SH_TOI_list = []
    EV_TOI_list = []
    Goals_list = []
    Assist_list = []
    rating_list = []
    pim_list = []
    shots_list = []
    hit_list = []
    gv_list = []
    tk_list = []
    bs_list = []
    fw_win_list = []
    fw_lose_list = []

    # Count the total nimber of players
    count_players = len(soup_summary.find_all('td', {'align': '', 'class': 'bborder + rborder'}))

    # Count of summary lines that we have gone through when gradually gathering info on each player
    summary_line_count = 0

    # Gathering info, one player at the time
    for i in range(0, count_players):

        # Detect if we have reaached summary line (TOI is empty)
        str_TOI = str(table_data[i * 22 + 6])
        str_rating = str(table_data[i * 22 + 3])
        if regex.findall(str_TOI)[0] == '\xa0' and regex.findall(str_rating)[0] != '\xa0':
            summary_line_count += 1
        str_name = str(soup_summary.find_all('td', {'align': '', 'class': 'bborder + rborder'})[i])
        # NEW
        if regex.findall(str_name)[0] == 'TEAM PENALTY':
            summary_line_count -= 1
        #############

        # Exctract important info
        name_dirty = soup_summary.find_all('td', {'align': '', 'class': 'bborder + rborder'})[i]
        toi_dirty = table_data[(i + summary_line_count) * 22 + 6]
        pp_toi_dirty = table_data[(i + summary_line_count) * 22 + 9]
        sh_toi_dirty = table_data[(i + summary_line_count) * 22 + 10]
        ev_toi_dirty = table_data[(i + summary_line_count) * 22 + 11]
        # 
        Goals_dirty = table_data[(i + summary_line_count) * 22]
        Assist_dirty = table_data[(i + summary_line_count) * 22 + 1]
        rating_dirty = table_data[(i + summary_line_count) * 22 + 3]
        pim_dirty = table_data[(i + summary_line_count) * 22 + 5]
        shots_dirty = table_data[(i + summary_line_count) * 22 + 12]
        hit_dirty = table_data[(i + summary_line_count) * 22 + 15]
        gv_dirty = table_data[(i + summary_line_count) * 22 + 16]
        tk_dirty = table_data[(i + summary_line_count) * 22 + 17]
        bs_dirty = table_data[(i + summary_line_count) * 22 + 18]
        fw_win_dirty = table_data[(i + summary_line_count) * 22 + 19]
        fw_lose_dirty = table_data[(i + summary_line_count) * 22 + 20]

        # Clean and add to list
        player_name_list.append(regex.findall(str(name_dirty))[0])
        TOI_list.append(regex.findall(str(toi_dirty))[0])
        PP_TOI_list.append(regex.findall(str(pp_toi_dirty))[0])
        SH_TOI_list.append(regex.findall(str(sh_toi_dirty))[0])
        EV_TOI_list.append(regex.findall(str(ev_toi_dirty))[0])
        #
        Goals_list.append(regex.findall(str(Goals_dirty))[0])
        Assist_list.append(regex.findall(str(Assist_dirty))[0])
        rating_list.append(regex.findall(str(rating_dirty))[0])
        pim_list.append(regex.findall(str(pim_dirty))[0])
        shots_list.append(regex.findall(str(shots_dirty))[0])
        hit_list.append(regex.findall(str(hit_dirty))[0])
        gv_list.append(regex.findall(str(gv_dirty))[0])
        tk_list.append(regex.findall(str(tk_dirty))[0])
        bs_list.append(regex.findall(str(bs_dirty))[0])
        fw_win_list.append(regex.findall(str(fw_win_dirty))[0])
        fw_lose_list.append(regex.findall(str(fw_lose_dirty))[0])

    results = pd.DataFrame({'player_name': player_name_list,
                            'TOI': TOI_list,
                            'PP_TOI': PP_TOI_list,
                            'SH_TOI': SH_TOI_list,
                            'EV_TOI': EV_TOI_list,
                            #
                            'Goals': Goals_list,
                            'Assists': Assist_list,
                            'Rating': rating_list,
                            'PIM': pim_list,
                            'Shots': shots_list,
                            'Hits': hit_list,
                            'Giveaways': gv_list,
                            'Takeaways': tk_list,
                            'Blocked shots': bs_list,
                            'FO wins': fw_win_list,
                            'FO losses': fw_lose_list
                            })
    results

    # Remove rows that do not correspond to a player
    results = results[results['player_name'] != 'TEAM PENALTY'].reset_index().drop(columns=['index'])

    ###########################################################################
    # Additional details
    ###########################################################################

    # Adding player number
    player_number_list = []
    number_soup = soup_summary.find_all('td', {'align': 'center', 'class': 'lborder + bborder + rborder'})

    for number_entry in number_soup:
        player_number_list.append(regex.findall(str(number_entry))[0])

    results['player_number'] = player_number_list

    # Adding player position
    player_position_list = []
    position_soup = soup_summary.find_all('td', {'align': 'center', 'class': 'bborder + rborder'})

    for position_entry in position_soup:
        position = regex.findall(str(position_entry))[0]
        if position in ['D', 'L', 'R', 'C', 'G']:
            player_position_list.append(position)

    results['position'] = player_position_list

    # Adding player team
    player_team_list = []
    cumm_goalie_count = 0

    for i in range(0, len(results.position)):

        if cumm_goalie_count >= 2:
            player_team_list.append(home_team_name)
        else:
            player_team_list.append(away_team_name)
        # First two goalies belong to away team
        if results.position[i] == 'G':
            cumm_goalie_count += 1

    results['team'] = player_team_list

    ###########################################################################
    # Create game ID
    ###########################################################################

    season_strings = str(year) + str(year + 1)

    # Add new column
    results['year_game_home_away'] = str(
        str(season_strings) + '-' + str(game_number) + '-' + home_team_name + '-' + away_team_name)

    return results
