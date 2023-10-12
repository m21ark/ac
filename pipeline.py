import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from clean import *


def pipeline_year(year=10):
    # Load the datasets
    df_teams = pandas.read_csv("dataset/teams.csv")
    df_teams_post = pandas.read_csv("dataset/teams_post.csv")
    df_series_post = pandas.read_csv("dataset/series_post.csv")
    df_players = pandas.read_csv("dataset/players.csv")
    df_players_teams = pandas.read_csv("dataset/players_teams.csv")
    df_coaches = pandas.read_csv("dataset/coaches.csv")
    df_awards_players = pandas.read_csv("dataset/awards_players.csv")

    dfs = [df_teams, df_teams_post, df_series_post, df_players,
           df_players_teams, df_coaches, df_awards_players]
    dfs_names = ["teams", "teams_post", "series_post",
                 "players", "players_teams", "coaches", "awards_players"]

    df_players_teams = clean_teams_players(df_players_teams)
    df_players = clean_players(df_players)

    df_merged = merge_player_info(df_players, df_players_teams)

    # collumn tmID and stint should be dropped
    df_players_teams = df_players_teams.drop(['tmID', 'stint'], axis=1)

    df_player_ratings = player_rankings(df_merged, year=year-1)

    df_players_teams = player_in_team_by_year(df_merged)

    df_players_teams = team_mean(df_players_teams, df_player_ratings)

    # call model

    # print(df_players_teams, df_teams)
    # call classification method
    df_teams = classify_playoff_entry(df_players_teams, df_teams, year)

    # call evaluation function

    return df_teams


def classify_playoff_entry(df_players_teams, df_teams, year):
    df_teams_at_year = df_players_teams[df_players_teams.year == year]

    # add to the df_teams at year the division from df_teams
    df_teams_at_year = df_teams_at_year.merge(
        df_teams[['tmID', 'year', 'confID']], on=['tmID', 'year'], how='left')

    ea_conf = df_teams_at_year[df_teams_at_year.confID == "EA"]
    we_conf = df_teams_at_year[df_teams_at_year.confID == "WE"]

    ea_conf = ea_conf.sort_values(by=['mean'], ascending=False)
    we_conf = we_conf.sort_values(by=['mean'], ascending=False)

    df_teams['playoff'] = "N"

    ea_playoff_teams = ea_conf.head(4)
    we_playoff_teams = we_conf.head(4)

    print(ea_playoff_teams)
    print(we_playoff_teams)

    for index, row in ea_playoff_teams.iterrows():
        row['tmID'] = 'Y'

    for index, row in we_playoff_teams.iterrows():
        row['tmID'] = 'Y'

    return df_teams


def merge_player_info(df_players, df_players_teams):
    df_merged = df_players_teams.merge(
        df_players, left_on='playerID', right_on='bioID')

    df_merged = df_merged.drop(['bioID'], axis=1)

    return df_merged


def player_in_team_by_year(df_players_teams):
    return df_players_teams.groupby(['tmID', 'year'])['playerID'].agg(list).reset_index()


def team_mean(df_players_teams, df_pred):
    mean_l = []

    for i in df_players_teams['playerID']:
        # get the 12 best players from the team
        top_12_players = df_pred[df_pred['playerID'].isin(i)].head(12)

        # add the mean to the team
        mean_l.append(top_12_players['predictions'].mean())

    df_players_teams['mean'] = mean_l
    return df_players_teams


def player_ranking_evolution(df_merged, playerID):
    df_merged = df_merged[df_merged['playerID'] == playerID]

    df_merged = df_merged.groupby(['playerID', 'year']).agg({
        'Points': 'sum',
        'TotalMinutes': 'sum',
        'TotaloRebounds': 'sum',
        'TotaldRebounds': 'sum',
        'TotalRebounds': 'sum',
        'TotalAssists': 'sum',
        'TotalSteals': 'sum',
        'TotalBlocks': 'sum',
        'TotalTurnovers': 'sum',
        'TotalPF': 'sum',
        'TotalfgAttempted': 'sum',
        'TotalfgMade': 'sum',
        'TotalftAttempted': 'sum',
        'TotalftMade': 'sum',
        'TotalthreeAttempted': 'sum',
        'TotalthreeMade': 'sum',
        'TotalGP': 'sum'
    })

    df_merged = df_merged.reset_index()

    df_merged = df_merged.drop(['TotalMinutes', 'TotalGP'], axis=1)

    players_id = df_merged['playerID']
    year = df_merged['year']

    gb_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # drop all non numerical atributes
    df_merged = df_merged.select_dtypes(include=['float64', 'int64'])

    df_merged = df_merged.drop(['year'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_merged)

    df_merged = pandas.DataFrame(X_scaled, columns=df_merged.columns)

    # TODO: make this not by year but by player years performance
    df_merged['medium'] = df_merged.mean(axis=1)

    Y_ = df_merged['medium']

    gb_model.fit(X_scaled, Y_)

    y_pred = gb_model.predict(X_scaled)

    mse = mean_squared_error(Y_, y_pred)

    # add player Id back and print the predictions for each row
    df_merged['playerID'] = players_id
    df_merged['predictions'] = y_pred
    df_merged['medium'] = Y_
    df_merged['year'] = year

    # drop all atributes exept playerID, medium and predictions
    df_pred = df_merged[['year', 'playerID', 'medium', 'predictions']]

    return df_pred


def player_rankings(df_merged, year=10):

    df_merged = df_merged[df_merged['year'] <= year]

    # df_merged['PointsPerMin'] = df_merged['Points'].div(df_merged['TotalMinutes'])
    # df_merged['oReboundsPerMin'] = df_merged['TotaloRebounds'].div(df_merged['TotalMinutes'])
    # df_merged['dReboundsPerMin'] = df_merged['TotaldRebounds'].div(df_merged['TotalMinutes'])
    # df_merged['ReboundsPerMin'] = df_merged['TotalRebounds'].div(df_merged['TotalMinutes'])
    # df_merged['AssistsPerMin'] = df_merged['TotalAssists'].div(df_merged['TotalMinutes'])
    # df_merged['StealsPerMin'] = df_merged['TotalSteals'].div(df_merged['TotalMinutes'])
    # df_merged['BlocksPerMin'] = df_merged['TotalBlocks'].div(df_merged['TotalMinutes'])
    # df_merged['TurnoversPerMin'] = df_merged['TotalTurnovers'].div(df_merged['TotalMinutes'])
    # df_merged['PFPerMin'] = df_merged['TotalPF'].div(df_merged['TotalMinutes'])
    # df_merged['fgAttemptedPerMin'] = df_merged['TotalfgAttempted'].div(df_merged['TotalMinutes'])
    # df_merged['fgMadePerMin'] = df_merged['TotalfgMade'].div(df_merged['TotalMinutes'])
    # df_merged['ftAttemptedPerMin'] = df_merged['TotalftAttempted'].div(df_merged['TotalMinutes'])
    # df_merged['ftMadePerMin'] = df_merged['TotalftMade'].div(df_merged['TotalMinutes'])
    # df_merged['threeAttemptedPerMin'] = df_merged['TotalthreeAttempted'].div(df_merged['TotalMinutes'])
    # df_merged['threeMadePerMin'] = df_merged['TotalthreeMade'].div(df_merged['TotalMinutes'])
    #
    # df_merged.fillna(0, inplace=True)
    # Existe um problema com esta abordagem, jogadores que jogaram pouco mas bem podem ter uma vantagem ... TODO: testar com isto
    df_merged = df_merged.groupby(['playerID', 'year']).agg({
        'Points': 'sum',
        'TotalMinutes': 'sum',
        'TotaloRebounds': 'sum',
        'TotaldRebounds': 'sum',
        'TotalRebounds': 'sum',
        'TotalAssists': 'sum',
        'TotalSteals': 'sum',
        'TotalBlocks': 'sum',
        'TotalTurnovers': 'sum',
        'TotalPF': 'sum',
        'TotalfgAttempted': 'sum',
        'TotalfgMade': 'sum',
        'TotalftAttempted': 'sum',
        'TotalftMade': 'sum',
        'TotalthreeAttempted': 'sum',
        'TotalthreeMade': 'sum',
        'TotalGP': 'sum'
    })

    df_merged = df_merged.reset_index()

    # group by year and player
    df_merged = df_merged.groupby(['playerID']).agg({
        'Points': 'mean',
        'TotalMinutes': 'mean',
        'TotaloRebounds': 'mean',
        'TotaldRebounds': 'mean',
        'TotalRebounds': 'mean',
        'TotalAssists': 'mean',
        'TotalSteals': 'mean',
        'TotalBlocks': 'mean',
        'TotalTurnovers': 'mean',
        'TotalPF': 'mean',
        'TotalfgAttempted': 'mean',
        'TotalfgMade': 'mean',
        'TotalftAttempted': 'mean',
        'TotalftMade': 'mean',
        'TotalthreeAttempted': 'mean',
        'TotalthreeMade': 'mean',
        'TotalGP': 'mean'
    })
    df_merged = df_merged.reset_index()

    # todo: tirar de cima isto
    df_merged = df_merged.drop(['TotalMinutes', 'TotalGP'], axis=1)

    players_id = df_merged['playerID']

    gb_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # drop all non numerical atributes
    df_merged = df_merged.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_merged)

    df_merged = pandas.DataFrame(X_scaled, columns=df_merged.columns)

    # TODO: make this not by year but by player years performance
    df_merged['medium'] = df_merged.mean(axis=1)

    Y_ = df_merged['medium']

    gb_model.fit(X_scaled, Y_)

    y_pred = gb_model.predict(X_scaled)

    mse = mean_squared_error(Y_, y_pred)

    # add player Id back and print the predictions for each row
    df_merged['playerID'] = players_id
    df_merged['predictions'] = y_pred
    df_merged['medium'] = Y_

    # drop all atributes exept playerID, medium and predictions
    df_pred = df_merged[['playerID', 'medium', 'predictions']]

    return df_pred


df = pipeline_year(10)
# print(df)

# df = pandas.read_csv("./dataset/coaches.csv")

# print(df.head(), end="\n\n\n")
# df = clean_coaches(df)
# print(df.head())
