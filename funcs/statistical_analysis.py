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

    ea_conf = ea_conf.sort_values(by=['predictions'], ascending=False)
    we_conf = we_conf.sort_values(by=['predictions'], ascending=False)

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


def team_mean(df_players_teams, df_pred, df_offensive_player_stats, df_defensive_player_stats):
    mean_l = []
    mean_o = []
    mean_d = []
    for i in df_players_teams['playerID']:
        # get the 12 best players from the team
        top_12_players = df_pred[df_pred['playerID'].isin(i)].sort_values(by=['predictions'], ascending=False).head(12)

        # add the mean to the team
        mean_l.append(top_12_players['predictions'].mean())

        top_12_offensive_players = df_offensive_player_stats[df_offensive_player_stats['playerID']
                                                             .isin(i)].sort_values(by=['predictions'], ascending=False).head(12)

        mean_o.append(top_12_offensive_players['predictions'].mean())

        top_12_defensive_players = df_defensive_player_stats[df_defensive_player_stats['playerID']
                                                             .isin(i)].sort_values(by=['predictions'], ascending=False).head(12)
        
        mean_d.append(top_12_defensive_players['predictions'].mean())

    #df_players_teams['mean'] = mean_l
    #df_players_teams['offensive_strength'] = mean_o
    #df_players_teams['defensive_strength'] = mean_d
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

def coach_ranking(df_coaches, year):

     df_coaches = df_coaches[df_coaches['year'] < year]

     df_coaches['win_percentage_coach'] = (df_coaches['won'] + df_coaches['post_wins']) / (df_coaches['won'] +
                                                                                                                  df_coaches['lost'] + df_coaches['post_wins'] + df_coaches['post_losses'])
     df_coach_stats = df_coaches.groupby(['coachID']).agg({
         'win_percentage_coach': 'mean',
     })
     df_coach_stats = df_coach_stats.reset_index()

     #divide win_percentage_coach by the number of years
     #df_coach_stats['win_percentage_coach'] = df_coach_stats['win_percentage_coach'] / (year - 1)

     # use standard scaler to scale the data
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(df_coach_stats.select_dtypes(include=[float]))

     df_coach_stats_scaled = pandas.DataFrame(X_scaled, columns=df_coach_stats.select_dtypes(include=[float]).columns)

     # restore the string features after scaling
     df_coach_stats_scaled[['coachID']] = df_coach_stats[['coachID']]
     #df_coach_stats_scaled[['win_percentage_coach']] = df_coach_stats[['win_percentage_coach']]

     return df_coach_stats_scaled


def player_rankings(df_merged, year=10): # note : year is the last year to count

    df_merged = df_merged[df_merged['year'] <= year]
    # df_merged = df_merged[df_merged['year'] >= year-4]

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


def team_rankings(df_teams, year=10): # note : year is the last year to count
    df_teams = df_teams[df_teams['year'] <= year] # The year results of the year can't be included

    # Make a copy of the df_teams with the year, tmID, confID and playoff columns
    df_teams_save = df_teams[['tmID', 'year', 'confID', 'playoff']]

    df_teams = df_teams.drop(['confID', 'playoff'], axis=1)
    df_teams.loc[:, 'year'] = df_teams['year'].astype(int) + 1

    columns_to_standardize = ['attend']
    

    # Standardize some of the columns only
    df_teams.loc[:,columns_to_standardize] = StandardScaler().fit_transform(df_teams[columns_to_standardize])

    # Add columns
    df_teams.loc[:,'team_stats'] = df_teams[columns_to_standardize].mean(axis=1).values

    # Drop columns of stats
    df_teams = df_teams.drop(columns_to_standardize, axis=1)

    # Drop columns that are not needed for now, but might be used in the future if considered relevant
    df_teams = df_teams.drop(['homeWinPercentage', 'awayWinPercentage', 'min'], axis=1)

    df_teams = df_teams_save.merge(df_teams, left_on=['tmID', 'year'], right_on=['tmID', 'year'], how='left')

    df_teams['RoundReached'] = df_teams['RoundReached'].fillna(0)
    df_teams['winPercentage'] = df_teams['winPercentage'].fillna(0)
    df_teams['team_stats'] = df_teams['team_stats'].fillna(0)

    return df_teams
# make a player offensive rating 

def player_offensive_rating(df_merged, year=10): # note : year is the last year to count
    
    df_merged = df_merged[df_merged['year'] <= year]
    # df_merged = df_merged[df_merged['year'] >= year-4]

    df_merged = df_merged.groupby(['playerID', 'year']).agg({
        'Points': 'sum',
        'TotalMinutes': 'sum',
        'TotaloRebounds': 'sum',
        'TotalAssists': 'sum',
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
        'Points': 'sum',
        'TotalMinutes': 'sum',
        'TotaloRebounds': 'sum',
        'TotalAssists': 'sum',
        'TotalfgAttempted': 'sum',
        'TotalfgMade': 'sum',
        'TotalftAttempted': 'sum',
        'TotalftMade': 'sum',
        'TotalthreeAttempted': 'sum',
        'TotalthreeMade': 'sum',
        'TotalGP': 'sum'
    })
    df_merged = df_merged.reset_index()

    # todo: tirar de cima isto
    df_merged = df_merged.drop(['TotalMinutes', 'TotalGP'], axis=1)

    players_id = df_merged['playerID']

    # drop all non numerical atributes
    df_merged = df_merged.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_merged)

    df_merged = pandas.DataFrame(X_scaled, columns=df_merged.columns)

    # TODO: make this not by year but by player years performance
    df_merged['medium'] = df_merged.mean(axis=1)

    Y_ = df_merged['medium']

    # add player Id back and print the predictions for each row
    df_merged['playerID'] = players_id
    df_merged['predictions'] = Y_
    df_merged['medium'] = Y_

    # drop all atributes exept playerID, medium and predictions
    df_pred = df_merged[['playerID', 'medium', 'predictions']]

    return df_pred

def add_player_stats(df_merged, year = 10):
    df_merged = df_merged[df_merged['year'] <= year]
    # df_merged = df_merged[df_merged['year'] >= year-4]

    df_merged = df_merged.groupby(['playerID', 'year', 'tmID']).agg({
        'Points': 'mean',
        'TotaloRebounds': 'mean',
        'TotaldRebounds': 'mean',
        'TotalAssists': 'mean',
        'TotalSteals': 'mean',
        'TotalBlocks': 'mean',
        'TotalTurnovers': 'mean',
        'TotalPF': 'mean',
        'TotalfgMade': 'mean',
        'TotalftMade': 'mean',
        'TotalthreeMade': 'mean',
    })

    df_merged = df_merged.reset_index()

    # group by year and player
    # df_merged = df_merged.groupby(['playerID']).agg({
        # 'Points': 'mean',
        # 'TotalMinutes': 'mean',
        # 'TotaloRebounds': 'mean',
        # 'TotaldRebounds': 'mean',
        # 'TotalRebounds': 'mean',
        # 'TotalAssists': 'mean',
        # 'TotalSteals': 'mean',
        # 'TotalBlocks': 'mean',
        # 'TotalTurnovers': 'mean',
        # 'TotalPF': 'mean',
        # 'TotalfgAttempted': 'mean',
        # 'TotalfgMade': 'mean',
        # 'TotalftAttempted': 'mean',
        # 'TotalftMade': 'mean',
        # 'TotalthreeAttempted': 'mean',
        # 'TotalthreeMade': 'mean',
        # 'TotalGP': 'mean'
    # })
    # df_merged = df_merged.reset_index()

    return df_merged


def defensive_player_ranking(df_merged, year=10): # note : year is the last year to count
    
    df_merged = df_merged[df_merged['year'] <= year]
    # df_merged = df_merged[df_merged['year'] >= year-4]

    df_merged = df_merged.groupby(['playerID', 'year']).agg({
        'TotalMinutes': 'sum',
        'TotaldRebounds': 'sum',
        'TotalSteals': 'sum',
        'TotalBlocks': 'sum',
        'TotalTurnovers': 'sum',
        'TotalPF': 'sum',
        'TotalGP': 'sum'
    })

    df_merged = df_merged.reset_index()

    # group by year and player
    df_merged = df_merged.groupby(['playerID']).agg({
        'TotalMinutes': 'sum',
        'TotaldRebounds': 'sum',
        'TotalSteals': 'sum',
        'TotalBlocks': 'sum',
        'TotalTurnovers': 'sum',
        'TotalPF': 'sum',
        'TotalGP': 'sum'
    })
    df_merged = df_merged.reset_index()

    # todo: tirar de cima isto
    df_merged = df_merged.drop(['TotalMinutes', 'TotalGP'], axis=1)

    players_id = df_merged['playerID']

    # drop all non numerical atributes
    df_merged = df_merged.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_merged)

    df_merged = pandas.DataFrame(X_scaled, columns=df_merged.columns)

    # TODO: make this not by year but by player years performance
    df_merged['medium'] = df_merged.mean(axis=1)

    Y_ = df_merged['medium']

    # add player Id back and print the predictions for each row
    df_merged['playerID'] = players_id
    df_merged['predictions'] = Y_
    df_merged['medium'] = Y_

    # drop all atributes exept playerID, medium and predictions
    df_pred = df_merged[['playerID', 'medium', 'predictions']]

    return df_pred
