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
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")


def unwrap_table_content(content):
    'Returns string from table format'
    left_index = str(content).find('">')
    right_index = str(content).find('</')
    info = str(content)[left_index + 2:right_index]
    return info


def get_salary_info(first, last):
    # Create player to query
    player_name = str(first) + '-' + str(last)

    # Remove unsuported characters
    player_name = player_name.replace('.', '')
    player_name = player_name.replace("'", '')
    player_name = player_name.replace('.', '')
    player_name = player_name.replace('Ü', 'U')
    player_name = player_name.replace('È', 'E')
    player_name = player_name.replace('É', 'E')

    # Get web page
    url = ("https://www.capfriendly.com/players/" + player_name.lower())
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    html = BeautifulSoup(webpage, "html.parser")

    # Extract info on active years (in table format)
    years = html.find_all('td', {'class': 'left'})
    player_details = html.find_all('td', {'class': 'center'})

    season, clause, cap_hit, aav, p_bonuses = [], [], [], [], []
    s_bonuses, base_salary, total_salary, minors_sal = [], [], [], []

    # Entry level slide counter (see Laurent Dauphin page)
    entry_counter = 0

    # Create table with info
    for i in range(0, len(years)):
        season.append(unwrap_table_content(years[i]))
        clause.append(unwrap_table_content(player_details[i * 8 - entry_counter]))
        cap_hit.append(unwrap_table_content(player_details[i * 8 + 1 - entry_counter]))
        aav.append(unwrap_table_content(player_details[i * 8 + 2 - entry_counter]))
        p_bonuses.append(unwrap_table_content(player_details[i * 8 + 3 - entry_counter]))
        s_bonuses.append(unwrap_table_content(player_details[i * 8 + 4 - entry_counter]))
        base_salary.append(unwrap_table_content(player_details[i * 8 + 5 - entry_counter]))
        if 'ENTRY-LEVEL SLIDE' in str(player_details[i * 8 + 5 - entry_counter]):
            entry_counter += 2
            total_salary.append('ENTRY-LEVEL SLIDE')
            minors_sal.append('ENTRY-LEVEL SLIDE')
        else:
            total_salary.append(unwrap_table_content(player_details[i * 8 + 6 - entry_counter]))
            minors_sal.append(unwrap_table_content(player_details[i * 8 + 7 - entry_counter]))

    df = pd.DataFrame({
        'season': season,
        'clause': clause,
        'cap_hit': cap_hit,
        'aav': aav,
        'p_bonuses': p_bonuses,
        's_bonuses': s_bonuses,
        'base_salary': base_salary,
        'total_salary': total_salary,
        'minors_sal': minors_sal})

    # Determine if year column is actually year
    df['year_test'] = list(map(lambda x: x[0:2], df['season']))

    # Return only pertinent columns
    final = df[df['year_test'] == '20']
    final = final.drop(['year_test'], axis=1)
    final['player_name'] = player_name

    return final


tough_names = ['J.T. MILLER',
               "K'ANDRE MILLER",
               'P.K. SUBBAN',
               "RYAN O'REILLY",
               'TIM STÜTZLE',
               "LOGAN O'CONNOR",
               'ALEXIS LAFRENIÈRE',
               'J.T. COMPHER',
               'T.J. OSHIE',
               "LIAM O'BRIEN",
               "DREW O'CONNOR",
               'ALEX BARRÉ-BOULET',
               'T.J. TYNAN',
               "DANNY O'REGAN",
               'A.J. GREER',
               'C.J. SMITH']
