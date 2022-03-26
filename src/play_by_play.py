import re

import numpy as np
import pandas as pd
import regex as re
import requests
from bs4 import BeautifulSoup


def extract_html_game_plays(year, game_number):
    ''' Takes the season and game number, scrapes the NHL website and returns html stats'''

    season_strings = str(year) + str(year + 1)

    game_number_string = str(game_number)
    if len(game_number_string) == 1:
        game_number_string = str(str('000') + str(game_number))
    elif len(game_number_string) == 2:
        game_number_string = str(str('00') + str(game_number))
    elif len(game_number_string) == 3:
        game_number_string = str(str('0') + str(game_number))

    full_web_address = "http://www.nhl.com/scores/htmlreports/" + season_strings + \
                       "/PL02" + game_number_string + ".HTM"

    # Extract data
    page_play_by_play = requests.get(full_web_address)
    soup_play_by_play = BeautifulSoup(page_play_by_play.content, 'html.parser')

    # Remove inconsistent spacing breaks
    removal = soup_play_by_play.findAll('td', {'align': 'center', 'class': ['heading + bborder']})
    removal
    for match in removal:
        match.decompose()

    # Remove new page breaks from play by play data
    removal_2 = soup_play_by_play.findAll('td', {'align': 'left', 'class': ['heading + bborder'], 'width': '58%'})
    removal_2
    for match in removal_2:
        match.decompose()

    # Separate all items remaining
    play_by_play_details = soup_play_by_play.find_all('td', {'class': 'bborder'})

    # Extract home and away teams
    teams = soup_play_by_play.find_all('td', {'align': 'center', 'style': 'font-size: 10px;font-weight:bold'})
    regex = re.compile(r'>(.*)<br/?>')
    home_team = regex.findall(str(teams[7]))[0]
    away_team = regex.findall(str(teams[0]))[0]

    return play_by_play_details, home_team, away_team


def game_play_to_pd(play_by_play_details):
    ''' Create pd from html play by play data'''

    ###################################################
    # Finding right first and last row of data
    ###################################################

    #### Ensure that we find first row with player info (skip anthems, etc)
    first_row = []

    for i in range(0, len(play_by_play_details)):
        # Look at 6th column: players for home team
        if i % 8 - 6 == 0:

            # Count number of players on line
            num_players = len(play_by_play_details[i].find_all('font'))
            # print(num_players)

            # Identify the first line where there are 6 players
            if (num_players == 6):
                first_row.append(i)

    first_row = np.min(first_row) - 6

    ### Ensure that we find last row with game info or before shootout
    last_row = []

    for i in range(0, len(play_by_play_details)):
        # Look at column 1, with the period number
        if i % 8 - 1 == 0:
            # Stop if we reach period 5 (shoot-out)
            period_string = str(play_by_play_details[i])
            first_index = period_string.find('>') + 1
            second_index = period_string.find('</td')
            period_final = period_string[first_index:second_index]
            if (period_final == '5'):
                last_row.append(i - 2)

            # Stop if we reach game end
            string_event_description = str(play_by_play_details[i + 3])
            first_index = string_event_description.find('>') + 1
            second_index = string_event_description.find('</')
            substring_event_description = string_event_description[first_index:second_index]
            if (substring_event_description == "GEND"):
                last_row.append(i - 2)

            # Because there are situations where there is no period 5 and no GEND, find last row
        last_row.append(len(play_by_play_details) - 1)

    last_row = np.min(last_row)

    ###################################################
    # Loop through all html code
    ###################################################

    # Declare lists
    action_number = []
    period_number = []
    strength = []
    action_start = []
    action_end = []
    action_description = []
    action_details = []
    away_player_1_name = []
    away_player_1_position = []
    away_player_1_number = []
    away_player_2_name = []
    away_player_2_position = []
    away_player_2_number = []
    away_player_3_name = []
    away_player_3_position = []
    away_player_3_number = []
    away_player_4_name = []
    away_player_4_position = []
    away_player_4_number = []
    away_player_5_name = []
    away_player_5_position = []
    away_player_5_number = []
    away_player_6_name = []
    away_player_6_position = []
    away_player_6_number = []
    home_player_1_name = []
    home_player_1_position = []
    home_player_1_number = []
    home_player_2_name = []
    home_player_2_position = []
    home_player_2_number = []
    home_player_3_name = []
    home_player_3_position = []
    home_player_3_number = []
    home_player_4_name = []
    home_player_4_position = []
    home_player_4_number = []
    home_player_5_name = []
    home_player_5_position = []
    home_player_5_number = []
    home_player_6_name = []
    home_player_6_position = []
    home_player_6_number = []

    ###### Loop START

    play_by_play_details = play_by_play_details[first_row:last_row + 1]

    for i in range(0, len(play_by_play_details)):
        j = i
        if (j % 8 == 0):
            string_event_number = str(play_by_play_details[i])
            first_index = string_event_number.find('>') + 1
            second_index = string_event_number.find('</td')
            event_number_final = string_event_number[first_index:second_index]
            action_number.append(event_number_final)


        elif (j % 8 == 1):
            period_string = str(play_by_play_details[i])
            first_index = period_string.find('>') + 1
            second_index = period_string.find('</td')
            period_final = period_string[first_index:second_index]
            period_number.append(period_final)


        elif (j % 8 == 2):
            string_play_type = str(play_by_play_details[i])
            first_index = string_play_type.find('>') + 1
            second_index = string_play_type.find('</td')
            play = string_play_type[first_index:second_index]
            strength.append(play)


        elif (j % 8 == 3):
            string_time = str(play_by_play_details[i])

            # Start time
            first_index = string_time.find('>') + 1
            second_index = string_time.find('<br')
            time_one = string_time[first_index:second_index]
            action_start.append(time_one)

            # End time
            first_index_2 = string_time.find('/>') + 2
            second_index_2 = string_time.find('</td')
            time_two = string_time[first_index_2:second_index_2]
            action_end.append(time_two)


        elif (j % 8 == 4):
            string_event_description = str(play_by_play_details[i])
            first_index = string_event_description.find('>') + 1
            second_index = string_event_description.find('</')
            substring_event_description = string_event_description[first_index:second_index]
            action_description.append(substring_event_description)

        elif (j % 8 == 5):
            string_event_description = str(play_by_play_details[i])
            first_index = string_event_description.find('>') + 1
            second_index = string_event_description.find('</')
            substring_event_description = string_event_description[first_index:second_index]
            action_details.append(substring_event_description)

        elif (j % 8 == 6):
            number_of_players_away = len(play_by_play_details[i].find_all('font'))

            if (number_of_players_away < 4):

                away_player_1_name.append(" ")
                away_player_1_position.append(" ")
                away_player_1_number.append(" ")

                away_player_2_name.append(" ")
                away_player_2_position.append(" ")
                away_player_2_number.append(" ")
                away_player_3_name.append(" ")
                away_player_3_position.append(" ")
                away_player_3_number.append(" ")

                away_player_4_name.append(" ")
                away_player_4_position.append(" ")
                away_player_4_number.append(" ")

                away_player_5_name.append(" ")
                away_player_5_position.append(" ")
                away_player_5_number.append(" ")

                away_player_6_name.append(" ")
                away_player_6_position.append(" ")
                away_player_6_number.append(" ")

            else:
                all_players_away = play_by_play_details[i].find_all('font')

                string_1 = str(all_players_away[0])
                string_2 = str(all_players_away[1])
                string_3 = str(all_players_away[2])
                string_4 = str(all_players_away[3])

                # Away player 1
                name_index_1 = string_1.find('- ') + 2
                name_index_2 = string_1.find('">')
                position_index_1 = string_1.find('title="') + 7
                position_index_2 = string_1.find(' -')
                number_index_1 = string_1.find('>') + 1
                number_index_2 = string_1.find('</')

                away_player_1_name.append(string_1[name_index_1:name_index_2])
                away_player_1_position.append(string_1[position_index_1:position_index_2])
                away_player_1_number.append(string_1[number_index_1:number_index_2])

                # Away player 2
                name_index_1 = string_2.find('- ') + 2
                name_index_2 = string_2.find('">')
                position_index_1 = string_2.find('title="') + 7
                position_index_2 = string_2.find(' -')
                number_index_1 = string_2.find('>') + 1
                number_index_2 = string_2.find('</')

                away_player_2_name.append(string_2[name_index_1:name_index_2])
                away_player_2_position.append(string_2[position_index_1:position_index_2])
                away_player_2_number.append(string_2[number_index_1:number_index_2])

                # Away player 3
                name_index_1 = string_3.find('- ') + 2
                name_index_2 = string_3.find('">')
                position_index_1 = string_3.find('title="') + 7
                position_index_2 = string_3.find(' -')
                number_index_1 = string_3.find('>') + 1
                number_index_2 = string_3.find('</')

                away_player_3_name.append(string_3[name_index_1:name_index_2])
                away_player_3_position.append(string_3[position_index_1:position_index_2])
                away_player_3_number.append(string_3[number_index_1:number_index_2])

                # Away player 4
                name_index_1 = string_4.find('- ') + 2
                name_index_2 = string_4.find('">')
                position_index_1 = string_4.find('title="') + 7
                position_index_2 = string_4.find(' -')
                number_index_1 = string_4.find('>') + 1
                number_index_2 = string_4.find('</')

                away_player_4_name.append(string_4[name_index_1:name_index_2])
                away_player_4_position.append(string_4[position_index_1:position_index_2])
                away_player_4_number.append(string_4[number_index_1:number_index_2])

                if (number_of_players_away == 4):
                    away_player_5_name.append(" ")
                    away_player_5_position.append(" ")
                    away_player_5_number.append(" ")

                    away_player_6_name.append(" ")
                    away_player_6_position.append(" ")
                    away_player_6_number.append(" ")

                if (number_of_players_away == 5):
                    string_5 = str(all_players_away[4])

                    # Away player 5
                    name_index_1 = string_5.find('- ') + 2
                    name_index_2 = string_5.find('">')
                    position_index_1 = string_5.find('title="') + 7
                    position_index_2 = string_5.find(' -')
                    number_index_1 = string_5.find('>') + 1
                    number_index_2 = string_5.find('</')

                    away_player_5_name.append(string_5[name_index_1:name_index_2])
                    away_player_5_position.append(string_5[position_index_1:position_index_2])
                    away_player_5_number.append(string_5[number_index_1:number_index_2])

                    away_player_6_name.append(" ")
                    away_player_6_position.append(" ")
                    away_player_6_number.append(" ")

                if (number_of_players_away >= 6):
                    string_5 = str(all_players_away[4])
                    string_6 = str(all_players_away[len(all_players_away) - 1])

                    # Away player 5
                    name_index_1 = string_5.find('- ') + 2
                    name_index_2 = string_5.find('">')
                    position_index_1 = string_5.find('title="') + 7
                    position_index_2 = string_5.find(' -')
                    number_index_1 = string_5.find('>') + 1
                    number_index_2 = string_5.find('</')

                    away_player_5_name.append(string_5[name_index_1:name_index_2])
                    away_player_5_position.append(string_5[position_index_1:position_index_2])
                    away_player_5_number.append(string_5[number_index_1:number_index_2])

                    # Away player 6
                    name_index_1 = string_6.find('- ') + 2
                    name_index_2 = string_6.find('">')
                    position_index_1 = string_6.find('title="') + 7
                    position_index_2 = string_6.find(' -')
                    number_index_1 = string_6.find('>') + 1
                    number_index_2 = string_6.find('</')

                    away_player_6_name.append(string_6[name_index_1:name_index_2])
                    away_player_6_position.append(string_6[position_index_1:position_index_2])
                    away_player_6_number.append(string_6[number_index_1:number_index_2])

        else:
            number_of_players_home = len(play_by_play_details[i].find_all('font'))

            if (number_of_players_home < 4):

                home_player_1_name.append(" ")
                home_player_1_position.append(" ")
                home_player_1_number.append(" ")

                home_player_2_name.append(" ")
                home_player_2_position.append(" ")
                home_player_2_number.append(" ")
                home_player_3_name.append(" ")
                home_player_3_position.append(" ")
                home_player_3_number.append(" ")

                home_player_4_name.append(" ")
                home_player_4_position.append(" ")
                home_player_4_number.append(" ")

                home_player_5_name.append(" ")
                home_player_5_position.append(" ")
                home_player_5_number.append(" ")

                home_player_6_name.append(" ")
                home_player_6_position.append(" ")
                home_player_6_number.append(" ")

            else:

                all_players_home = play_by_play_details[i].find_all('font')

                string_1 = str(all_players_home[0])
                string_2 = str(all_players_home[1])
                string_3 = str(all_players_home[2])
                string_4 = str(all_players_home[3])

                # HOME player 1
                name_index_1 = string_1.find('- ') + 2
                name_index_2 = string_1.find('">')
                position_index_1 = string_1.find('title="') + 7
                position_index_2 = string_1.find(' -')
                number_index_1 = string_1.find('>') + 1
                number_index_2 = string_1.find('</')

                home_player_1_name.append(string_1[name_index_1:name_index_2])
                home_player_1_position.append(string_1[position_index_1:position_index_2])
                home_player_1_number.append(string_1[number_index_1:number_index_2])

                # HOME player 2
                name_index_1 = string_2.find('- ') + 2
                name_index_2 = string_2.find('">')
                position_index_1 = string_2.find('title="') + 7
                position_index_2 = string_2.find(' -')
                number_index_1 = string_2.find('>') + 1
                number_index_2 = string_2.find('</')

                home_player_2_name.append(string_2[name_index_1:name_index_2])
                home_player_2_position.append(string_2[position_index_1:position_index_2])
                home_player_2_number.append(string_2[number_index_1:number_index_2])

                # HOME player 3
                name_index_1 = string_3.find('- ') + 2
                name_index_2 = string_3.find('">')
                position_index_1 = string_3.find('title="') + 7
                position_index_2 = string_3.find(' -')
                number_index_1 = string_3.find('>') + 1
                number_index_2 = string_3.find('</')

                home_player_3_name.append(string_3[name_index_1:name_index_2])
                home_player_3_position.append(string_3[position_index_1:position_index_2])
                home_player_3_number.append(string_3[number_index_1:number_index_2])

                # HOME player 4
                name_index_1 = string_4.find('- ') + 2
                name_index_2 = string_4.find('">')
                position_index_1 = string_4.find('title="') + 7
                position_index_2 = string_4.find(' -')
                number_index_1 = string_4.find('>') + 1
                number_index_2 = string_4.find('</')

                home_player_4_name.append(string_4[name_index_1:name_index_2])
                home_player_4_position.append(string_4[position_index_1:position_index_2])
                home_player_4_number.append(string_4[number_index_1:number_index_2])

                if (number_of_players_home == 4):
                    home_player_5_name.append(" ")
                    home_player_5_position.append(" ")
                    home_player_5_number.append(" ")

                    home_player_6_name.append(" ")
                    home_player_6_position.append(" ")
                    home_player_6_number.append(" ")

                if (number_of_players_home == 5):
                    string_5 = str(all_players_home[4])

                    # Away player 5
                    name_index_1 = string_5.find('- ') + 2
                    name_index_2 = string_5.find('">')
                    position_index_1 = string_5.find('title="') + 7
                    position_index_2 = string_5.find(' -')
                    number_index_1 = string_5.find('>') + 1
                    number_index_2 = string_5.find('</')

                    home_player_5_name.append(string_5[name_index_1:name_index_2])
                    home_player_5_position.append(string_5[position_index_1:position_index_2])
                    home_player_5_number.append(string_5[number_index_1:number_index_2])

                    home_player_6_name.append(" ")
                    home_player_6_position.append(" ")
                    home_player_6_number.append(" ")

                if (number_of_players_home >= 6):
                    string_5 = str(all_players_home[4])
                    string_6 = str(all_players_home[len(all_players_home) - 1])

                    # Away player 5
                    name_index_1 = string_5.find('- ') + 2
                    name_index_2 = string_5.find('">')
                    position_index_1 = string_5.find('title="') + 7
                    position_index_2 = string_5.find(' -')
                    number_index_1 = string_5.find('>') + 1
                    number_index_2 = string_5.find('</')

                    home_player_5_name.append(string_5[name_index_1:name_index_2])
                    home_player_5_position.append(string_5[position_index_1:position_index_2])
                    home_player_5_number.append(string_5[number_index_1:number_index_2])

                    # Away player 6
                    name_index_1 = string_6.find('- ') + 2
                    name_index_2 = string_6.find('">')
                    position_index_1 = string_6.find('title="') + 7
                    position_index_2 = string_6.find(' -')
                    number_index_1 = string_6.find('>') + 1
                    number_index_2 = string_6.find('</')

                    home_player_6_name.append(string_6[name_index_1:name_index_2])
                    home_player_6_position.append(string_6[position_index_1:position_index_2])
                    home_player_6_number.append(string_6[number_index_1:number_index_2])

                    ###### Loop END

    ###################################################
    # Create pd
    ###################################################

    df = pd.DataFrame({
        'Event': action_number,
        'Period': period_number,
        'Strength': strength,
        'Start': action_start,
        'End': action_end,
        'Description': action_description,
        'Details': action_details,
        'Away_p1_name': away_player_1_name,
        'Away_p1_pos': away_player_1_position,
        'Away_p1_num': away_player_1_number,
        'Away_p2_name': away_player_2_name,
        'Away_p2_pos': away_player_2_position,
        'Away_p2_num': away_player_2_number,
        'Away_p3_name': away_player_3_name,
        'Away_p3_pos': away_player_3_position,
        'Away_p3_num': away_player_3_number,
        'Away_p4_name': away_player_4_name,
        'Away_p4_pos': away_player_4_position,
        'Away_p4_num': away_player_4_number,
        'Away_p5_name': away_player_5_name,
        'Away_p5_pos': away_player_5_position,
        'Away_p5_num': away_player_5_number,
        'Away_p6_name': away_player_6_name,
        'Away_p6_pos': away_player_6_position,
        'Away_p6_num': away_player_6_number,
        'Home_p1_name': home_player_1_name,
        'Home_p1_pos': home_player_1_position,
        'Home_p1_num': home_player_1_number,
        'Home_p2_name': home_player_2_name,
        'Home_p2_pos': home_player_2_position,
        'Home_p2_num': home_player_2_number,
        'Home_p3_name': home_player_3_name,
        'Home_p3_pos': home_player_3_position,
        'Home_p3_num': home_player_3_number,
        'Home_p4_name': home_player_4_name,
        'Home_p4_pos': home_player_4_position,
        'Home_p4_num': home_player_4_number,
        'Home_p5_name': home_player_5_name,
        'Home_p5_pos': home_player_5_position,
        'Home_p5_num': home_player_5_number,
        'Home_p6_name': home_player_6_name,
        'Home_p6_pos': home_player_6_position,
        'Home_p6_num': home_player_6_number})

    return df


def get_game_play(year, game_number):
    ''' Takes the season and game number, scrapes the NHL website,
    takes html stats then returns pd'''

    season_strings = str(year) + str(year + 1)

    # Extract data from NHL site
    play_by_play, home_team_name, away_team_name = extract_html_game_plays(year, game_number)

    # Clean and create pd
    game_summary_pd = game_play_to_pd(play_by_play)

    # Add new column
    game_summary_pd['year_game_home_away'] = str(str(season_strings) + '-' + \
                                                 str(game_number) + '-' + \
                                                 home_team_name + '-' + \
                                                 away_team_name)
    return game_summary_pd
