import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def merge_player_info(df_players, df_players_teams):
    df_merged = df_players_teams.merge(df_players, left_on='playerID', right_on='bioID')

    df_merged = df_merged.drop(['bioID'], axis=1)

    return df_merged

def player_rankings(df_merged):
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
    
    
    #group by year and player
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
    
    
    #todo: tirar de cima isto
    df_merged = df_merged.drop(['TotalMinutes','TotalGP'], axis=1)

    players_id = df_merged['playerID']

    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # drop all non numerical atributes
    df_merged = df_merged.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_merged)

    df_merged = pandas.DataFrame(X_scaled, columns=df_merged.columns)


    df_merged['medium'] = df_merged.mean(axis=1) # TODO: make this not by year but by player years performance


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





def clean_teams_players(df_players_teams):
    df_players_teams = df_players_teams.loc[:,df_players_teams.nunique() > 1]

    df_players_teams = df_players_teams.drop(columns=[], axis=1)

    df_players_teams['TotalGP'] = df_players_teams.apply(lambda x: x['GP'] + x['PostGP'], axis=1)
    df_players_teams['Points'] = df_players_teams.apply(lambda x: x['PostPoints'] + x['points'], axis=1)
    df_players_teams['TotalMinutes'] = df_players_teams.apply(lambda x: x['minutes'] + x['PostMinutes'], axis=1)
    df_players_teams['TotaloRebounds'] = df_players_teams.apply(lambda x: x['oRebounds'] + x['PostoRebounds'], axis=1)
    df_players_teams['TotaldRebounds'] = df_players_teams.apply(lambda x: x['dRebounds'] + x['PostdRebounds'], axis=1)
    df_players_teams['TotalRebounds'] = df_players_teams.apply(lambda x: x['rebounds'] + x['PostRebounds'], axis=1)
    df_players_teams['TotalAssists'] = df_players_teams.apply(lambda x: x['assists'] + x['PostAssists'], axis=1)
    df_players_teams['TotalSteals'] = df_players_teams.apply(lambda x: x['steals'] + x['PostSteals'], axis=1)
    df_players_teams['TotalBlocks'] = df_players_teams.apply(lambda x: x['blocks'] + x['PostBlocks'], axis=1)
    df_players_teams['TotalTurnovers'] = df_players_teams.apply(lambda x: x['turnovers'] + x['PostTurnovers'], axis=1)
    df_players_teams['TotalPF'] = df_players_teams.apply(lambda x: x['PF'] + x['PostPF'], axis=1)
    df_players_teams['TotalfgAttempted'] = df_players_teams.apply(lambda x: x['fgAttempted'] + x['PostfgAttempted'], axis=1)
    df_players_teams['TotalfgMade'] = df_players_teams.apply(lambda x: x['fgMade'] + x['PostfgMade'], axis=1)
    df_players_teams['TotalftAttempted'] = df_players_teams.apply(lambda x: x['ftAttempted'] + x['PostftAttempted'], axis=1)
    df_players_teams['TotalftMade'] = df_players_teams.apply(lambda x: x['ftMade'] + x['PostftMade'], axis=1)
    df_players_teams['TotalthreeAttempted'] = df_players_teams.apply(lambda x: x['threeAttempted'] + x['PostthreeAttempted'], axis=1)
    df_players_teams['TotalthreeMade'] = df_players_teams.apply(lambda x: x['threeMade'] + x['PostthreeMade'], axis=1)
    df_players_teams['TotalGS'] = df_players_teams.apply(lambda x: x['GS'] + x['PostGS'], axis=1)
    df_players_teams['TotalDQ'] = df_players_teams.apply(lambda x: x['dq'] + x['PostDQ'], axis=1)



    df_players_teams = df_players_teams.drop(['GP', 'PostGP', 'points', 'PostPoints', 'minutes', 'PostMinutes', 'oRebounds', 'PostoRebounds', 'dRebounds', 'PostdRebounds', 'rebounds', 'PostRebounds', 'assists', 'PostAssists', 'steals', 'PostSteals', 'blocks', 'PostBlocks', 'turnovers', 'PostTurnovers', 'PF', 'PostPF', 'fgAttempted', 'PostfgAttempted', 'fgMade', 'PostfgMade', 'ftAttempted', 'PostftAttempted', 'ftMade', 'PostftMade', 'threeAttempted', 'PostthreeAttempted', 'threeMade', 'PostthreeMade', 'GS', 'PostGS', 'dq', 'PostDQ'], axis=1)


    return df_players_teams


def clean_players(df_players): 
    df_players = df_players.loc[:,df_players.nunique() > 1]
    df_players = df_players.drop(columns = ['college', 'collegeOther', 'birthDate', 'deathDate'], axis=1)


    return df_players


    


def remove_redundant_cols(df):
    return df.loc[:, df.nunique() > 1]


def replace_col_with_ids(df, col_name):
    unique_awards = df[col_name].unique()
    award_mapping = {award: i for i,
                     award in enumerate(unique_awards, start=1)}

    df[col_name] = df[col_name].replace(award_mapping)
    return df


def clean_awards_players(df):
    df = remove_redundant_cols(df)
    df = replace_col_with_ids(df, 'award')
    return df

def clean_coaches(df):
    df = remove_redundant_cols(df) # TODO: apply this func to all DFs
    return df

    


df = pandas.read_csv("./dataset/coaches.csv")

print(df.head(), end="\n\n\n")
df = clean_coaches(df)
print(df.head())
