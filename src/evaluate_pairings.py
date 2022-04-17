import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

# game_summaries_with_xG_agg = pd.read_csv('data/')
# game_summary_select_year = game_summaries_with_xG_agg[game_summaries_with_xG_agg['year']=='2021']


def create_summary(player_summary, shots, team_abb, position):

    if position != 'D':
        player_list = player_summary[(player_summary['team'] == team_abb) &
                                     (player_summary['pos'] != 'D') & (player_summary['pos'] != 'G')]
    else:
        player_list = player_summary[(player_summary['team'] == team_abb) &
                                     (player_summary['pos'] == 'D')]

    print(player_list.sort_values(by='number'))

    team_for = team_abb

    players_analyzed = []
    agg_summary = pd.DataFrame()

    for j in range(0, player_list.shape[0]):

        player_num = player_list.number.iloc[j]
        player_name = player_list.player.iloc[j]
        # print(player_name)

        if player_name not in players_analyzed:

            # Finding peers for a player
            pos = player_summary[(player_summary['team'] == team_for) &
                                 (player_summary['player'] == player_name)]['pos'].values[0]

            # print(pos)
            peers = player_summary[(player_summary['team'] == team_for)]
            # select peers with same position
            if pos in ['L', 'C', 'R']:
                peers = player_summary[(player_summary['team'] == team_for) &
                                       (player_summary['pos'] != 'D') &
                                       (player_summary['pos'] != 'G')]
                #print('here')
            else:
                peers = player_summary[(player_summary['team'] == team_for) &
                                       (player_summary['pos'] == 'D')]
                print('here2')

            # print(peers)

            # Shots for all games that montreal played, EV only
            shots_year = shots[(shots['Strength'] == 'EV')]
            # print(shots_year)
            shots_year['team_for_players'] = np.where(shots_year.team_for == shots_year.home_team_abb,
                                                      shots_year.home_num_array, shots_year.away_num_array)
            shots_year['team_against_players'] = np.where(shots_year.team_for == shots_year.away_team_abb,
                                                          shots_year.home_num_array, shots_year.away_num_array)
            shots_team_all = shots_year[(shots_year['away_team_abb'] == team_for) |
                                        (shots_year['home_team_abb'] == team_for)]

            # Plays for and against
            plays_for_subset = shots_team_all[shots_team_all['team_for'] == team_for]
            plays_against_subset = shots_team_all[shots_team_all['team_for'] != team_for]

            # Create lists to store data
            reference_player = []
            peer = []
            XG_for_with_ref = []
            XG_for_without_ref = []
            shots_for_with_ref = []
            shots_for_without_ref = []
            XG_against_with_ref = []
            XG_against_without_ref = []
            shots_against_with_ref = []
            shots_against_without_ref = []

            for unique_peer in peers.number.value_counts().index:
                # Filter only for plays that are for select reference player
                plays_for_subset['player_in'] = list(
                    map(lambda x: str(player_num) in x, plays_for_subset.team_for_players))
                plays_for_subset['peer_in'] = list(
                    map(lambda x: str(unique_peer) in x, plays_for_subset.team_for_players))

                plays_against_subset['player_in'] = list(
                    map(lambda x: str(player_num) in x, plays_against_subset.team_against_players))
                plays_against_subset['peer_in'] = list(
                    map(lambda x: str(unique_peer) in x, plays_against_subset.team_against_players))

                reference_player.append(player_num)
                peer.append(unique_peer)
                XG_for_with_ref.append(plays_for_subset[(plays_for_subset['player_in'] == True) &
                                                        (plays_for_subset['peer_in'] == True)].xG.sum())
                XG_for_without_ref.append(plays_for_subset[(plays_for_subset['player_in'] == False) &
                                                           (plays_for_subset['peer_in'] == True)].xG.sum())
                shots_for_with_ref.append(plays_for_subset[(plays_for_subset['player_in'] == True) &
                                                           (plays_for_subset['peer_in'] == True)].shape[0])
                shots_for_without_ref.append(plays_for_subset[(plays_for_subset['player_in'] == False) &
                                                              (plays_for_subset['peer_in'] == True)].shape[0])
                XG_against_with_ref.append(plays_against_subset[(plays_against_subset['player_in'] == True) &
                                                                (plays_against_subset['peer_in'] == True)].xG.sum())
                XG_against_without_ref.append(plays_against_subset[(plays_against_subset['player_in'] == False) &
                                                                   (plays_against_subset['peer_in'] == True)].xG.sum())
                shots_against_with_ref.append(plays_against_subset[(plays_against_subset['player_in'] == True) &
                                                                   (plays_against_subset['peer_in'] == True)].shape[0])
                shots_against_without_ref.append(plays_against_subset[(plays_against_subset['player_in'] == False) &
                                                                      (plays_against_subset['peer_in'] == True)].shape[0])

            summary = pd.DataFrame({'reference_player': reference_player,
                                    'peer': peer,
                                    'XG_for_with_ref': XG_for_with_ref,
                                    'XG_for_without_ref': XG_for_without_ref,
                                    'shots_for_with_ref': shots_for_with_ref,
                                    'shots_for_without_ref': shots_for_without_ref,
                                    'XG_against_with_ref': XG_against_with_ref,
                                    'XG_against_without_ref': XG_against_without_ref,
                                    'shots_against_with_ref': shots_against_with_ref,
                                    'shots_against_without_ref': shots_against_without_ref})

            summary['net_XG_with_ref'] = summary['XG_for_with_ref'] - summary['XG_against_with_ref']
            summary['net_XG_without_ref'] = summary['XG_for_without_ref'] - summary['XG_against_without_ref']

            summary['shots_with_ref'] = summary['shots_for_with_ref'] + summary['shots_against_with_ref']
            summary['shots_without_ref'] = summary['shots_for_without_ref'] + summary['shots_against_without_ref']

            summary['net_XG_with_ref_per_shot'] = summary['net_XG_with_ref'] / summary['shots_with_ref']
            summary['net_XG_without_ref_per_shot'] = summary['net_XG_without_ref'] / summary['shots_without_ref']

            summary['delta_with_ref'] = summary['net_XG_with_ref_per_shot'] - summary['net_XG_without_ref_per_shot']

            summary = summary[summary['reference_player'] != summary['peer']]

            summary = summary.sort_values(by='shots_with_ref', ascending=False)[
                ['reference_player', 'peer', 'shots_with_ref', 'shots_without_ref',
                 'net_XG_with_ref_per_shot', 'net_XG_without_ref_per_shot', 'delta_with_ref']]

            agg_summary = agg_summary.append(summary)

            players_analyzed.append(player_name)

    return agg_summary
