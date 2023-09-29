import pandas
import numpy as np

def merge_player_info(df_players, df_players_teams):
    df_merged = df_players_teams.merge(df_players, left_on='playerID', right_on='bioID')

    df_merged = df_merged.drop(['bioID'], axis=1)

    return df_merged

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
