import pandas as pd
import numpy as np

from funcs.statistical_analysis import *
from funcs.results_analysis import *
from funcs.merge import *


def load_data():
    # Load the clean datasets
    df_teams = pd.read_csv("dataset/cleaned/teams.csv")
    df_teams_post = pd.read_csv("dataset/cleaned/teams_post.csv")
    df_series_post = pd.read_csv("dataset/cleaned/series_post.csv")
    df_players = pd.read_csv("dataset/cleaned/players.csv")
    df_players_teams = pd.read_csv("dataset/cleaned/players_teams.csv")
    df_coaches = pd.read_csv("dataset/cleaned/coaches.csv")
    df_awards_players = pd.read_csv("dataset/cleaned/awards_players.csv")

    return [df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players]


def pipeline_year(year=10):

    # set all the dataframes
    df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players = load_data()

    df_awards_players, df_awards_coaches = separate_awards_info(
        df_awards_players, year)

    # Give the number of awards to players and coaches
    df_players = merge_awards_info(df_players, df_awards_players, year)
    df_coaches = merge_awards_info(df_coaches, df_awards_coaches, year)

    df_merged = merge_player_info(df_players, df_players_teams)

    # collumn tmID and stint should be dropped
    df_players_teams = df_players_teams.drop(['tmID', 'stint'], axis=1)

    print(df_merged)
    df_player_ratings = player_rankings(df_merged, year=year-1)

    df_players_teams = player_in_team_by_year(df_merged)

    df_players_teams = team_mean(df_players_teams, df_player_ratings)

    df_teams_merged = df_players_teams.merge(
        df_teams[['tmID', 'year', 'confID']], on=['tmID', 'year'], how='left')

    df_merged = merge_coach_info(df_teams_merged, df_coaches)

    df_teams, ea_teams, we_teams = classify_playoff_entry(
        df_teams_merged, year)

    ea_predictions = ea_teams['tmID'].unique()
    we_predictions = we_teams['tmID'].unique()

    calculate_playoff_accuracy(year, ea_predictions, we_predictions)

    return df_teams


if __name__ == "__main__":
    df = pipeline_year(10)
