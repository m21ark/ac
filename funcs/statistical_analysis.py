from colorama import Fore, Style
import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def classify_playoff_entry(df_teams, year):
    df_teams_at_year = df_teams[df_teams.year == year]

    ea_conf = df_teams_at_year[df_teams_at_year.confID == "EA"]
    we_conf = df_teams_at_year[df_teams_at_year.confID == "WE"]

    ea_conf = ea_conf.sort_values(by=['mean'], ascending=False)
    we_conf = we_conf.sort_values(by=['mean'], ascending=False)

    df_teams['playoff'] = "N"

    ea_playoff_teams = ea_conf.head(4)
    we_playoff_teams = we_conf.head(4)

    # print(ea_playoff_teams)
    # print(we_playoff_teams)

    for _, row in ea_playoff_teams.iterrows():
        row['playoff'] = 'Y'

    for _, row in we_playoff_teams.iterrows():
        row['playoff'] = 'Y'

    return df_teams, ea_playoff_teams, we_playoff_teams


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
        'TotalGP': 'sum',
        'award': 'sum',
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
