import pandas
import numpy as np


def clean_teams_players(df_players_teams):
    df_players_teams = df_players_teams.loc[:,df_players_teams.nunique() > 1]

    df_players_teams = df_players_teams.drop(columns=[], axis=1)

    return df_players_teams

#
#
#
# 
#
#
#

def clean_players(df_players): 
    df_players = df_players.loc[:,df_players.nunique() > 1]
    df_players = df_players.drop(columns = ['college', 'collegeOther', 'birthDate', 'deathDate'], axis=1)


    return df_players


    
