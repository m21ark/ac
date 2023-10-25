import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from funcs.statistical_analysis import *
from funcs.results_analysis import *
from funcs.merge import *
from funcs.clean import *


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


def apply_cleaning():

    # Loading Original DataFrames
    df_teams = pd.read_csv("dataset/original/teams.csv")
    df_teams_post = pd.read_csv("dataset/original/teams_post.csv")
    df_series_post = pd.read_csv("dataset/original/series_post.csv")
    df_players = pd.read_csv("dataset/original/players.csv")
    df_players_teams = pd.read_csv("dataset/original/players_teams.csv")
    df_coaches = pd.read_csv("dataset/original/coaches.csv")
    df_awards_players = pd.read_csv("dataset/original/awards_players.csv")

    # Cleaning DataFrames
    df_players = clean_players(df_players)
    df_awards_players = clean_awards_players(df_awards_players)
    df_coaches = clean_coaches(df_coaches)
    df_players_teams = clean_teams_players(df_players_teams)
    df_series_post = clean_series_post(df_series_post)
    df_teams_post = clean_teams_post(df_teams_post)
    df_teams = clean_teams(df_teams)

    # Saving DataFrames to CSV
    df_players.to_csv("dataset/cleaned/players.csv", index=False)
    df_awards_players.to_csv("dataset/cleaned/awards_players.csv", index=False)
    df_coaches.to_csv("dataset/cleaned/coaches.csv", index=False)
    df_players_teams.to_csv("dataset/cleaned/players_teams.csv", index=False)
    df_series_post.to_csv("dataset/cleaned/series_post.csv", index=False)
    df_teams_post.to_csv("dataset/cleaned/teams_post.csv", index=False)
    df_teams.to_csv("dataset/cleaned/teams.csv", index=False)


def pipeline_year(year=10, display_results=False):

    if year > 10 or year < 2:
        raise ValueError("Year must be between 2 and 10")

    # Load the clean datasets
    df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players = load_data()

    df_awards_players, df_awards_coaches = separate_awards_info(
        df_awards_players, year)

    # Give the number of awards to players and coaches
    df_players = merge_awards_info(df_players, df_awards_players, year)
    df_coaches = merge_awards_info(df_coaches, df_awards_coaches, year)
    df_merged = merge_player_info(df_players, df_players_teams)

    # collumn tmID and stint should be dropped
    df_players_teams = df_players_teams.drop(['tmID', 'stint'], axis=1)

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

    accuracy = calculate_playoff_accuracy(
        year, ea_predictions, we_predictions, display_results)

    return accuracy


def check_accuracy_by_year():
    accs = []
    for year in range(2, 11):
        acc = pipeline_year(year)
        accs.append(acc)

    # plot the accuracy line graph
    plt.plot(range(2, 11), accs, label="Accuracy")

    # add tags for Y on each X
    for i, acc in enumerate(accs):
        plt.text(i+2, acc, f"{acc:.2f}", ha="center", va="bottom")

    # add legend
    plt.legend()

    plt.xlabel("Year")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by year")
    plt.show()


if __name__ == "__main__":
    apply_cleaning()
    check_accuracy_by_year()
    # pipeline_year(5)