
import pandas as pd


# ======================== Utility functions ========================

def remove_redundant_cols(df):
    return df.loc[:, df.nunique() > 1]


def replace_col_with_ids(df, col_name):
    unique_awards = df[col_name].unique()
    award_mapping = {award: i for i,
                     award in enumerate(unique_awards, start=1)}

    df.loc[:, col_name] = df.loc[:, col_name].replace(award_mapping)
    return df


def basic_clean(df, cols=[]):
    df = df.drop(columns=cols, axis=1)
    df = remove_redundant_cols(df)
    return df


# ======================== Clean functions ========================

def clean_players(df):
    df = basic_clean(df, ['college', 'collegeOther', 'birthDate', 'deathDate'])
    return df


def clean_awards_players(df):
    df = basic_clean(df)
    return df


def clean_coaches(df):
    df = basic_clean(df)
    return df


def clean_series_post(df):
    df = basic_clean(df)
    return df


def clean_teams_post(df):
    df = basic_clean(df)
    return df


def clean_teams(df):
    df = basic_clean(df)

    df = df.drop(columns=['franchID',
                 'rank', 'name', "arena", "GP"], axis=1)
    df['confID'] = df['confID'].replace({'West': 0, 'East': 1})

    # Replace Playoffs with 1 and 0 instead of Y and N
    df['playoff'] = df['playoff'].replace('Y', 1)
    df['playoff'] = df['playoff'].replace('N', 0)

    # Replace df features firstRound, semis, final with a single feature named RoundReached with values 0, 1, 2, 3
    # 0 - not reached
    # 1 - first round
    # 2 - semi finals
    # 3 - finals
    df['firstRound'] = df['firstRound'].replace('W', 2)
    df['firstRound'] = df['firstRound'].replace('L', 1)
    df['firstRound'] = df['firstRound'].fillna(0)

    df['semis'] = df['semis'].replace('W', 1)
    df['semis'] = df['semis'].replace('L', 0)
    df['semis'] = df['semis'].fillna(0)

    df['finals'] = df['finals'].replace('W', 1)
    df['finals'] = df['finals'].replace('L', 0)
    df['finals'] = df['finals'].fillna(0)

    df['RoundReached'] = (df['firstRound'] + df['semis'] + df['finals']).astype(int)

    df = df.drop(columns=['firstRound', 'semis', 'finals'], axis=1)

    # Add a new feature that indicates the pecentage of games won
    df['winPercentage'] = round((df['homeW'] + df['awayW']) / (
        df['homeW'] + df['awayW'] + df['awayL'] + df['homeL']), 3)

    # Add a new feature that indicates the pecentage of home games won
    df['homeWinPercentage'] = round(df['homeW'] /
                                    (df['homeW'] + df['homeL']), 3)
    # Add a new feature that indicates the pecentage of away games won
    df['awayWinPercentage'] = round(df['awayW'] /
                                    (df['awayW'] + df['awayL']), 3)

    # Drop the W and L features
    df = df.drop(
        columns=['homeW', 'homeL', 'awayW', 'awayL', 'won', 'lost'], axis=1)

    # Remove confW and confL
    df = df.drop(columns=['confW', 'confL'], axis=1)

    # Calculate offensive ratio statistics (higher is better)
    df['of_goal'] = round(df['o_fgm'] / df['o_fga'], 2)
    df['of_3pt'] = round(df['o_3pm'] / df['o_3pa'], 2)
    df['of_throw'] = round(df['o_ftm'] / df['o_fta'], 2)
    df['of_reb'] = df['o_reb']
    df['of_assist'] = round(df['o_asts'] / df['o_to'], 2)

    # Calculate defensive ratio statistics (lower is better)
    df['df_goal'] = round(df['d_fgm'] / df['d_fga'], 2)
    df['df_3pt'] = round(df['d_3pm'] / df['d_3pa'], 2)
    df['df_throw'] = round(df['d_ftm'] / df['d_fta'], 2)
    df['df_reb'] = df['d_reb']
    df['df_steal'] = df['d_stl']

    # Drop the columns you no longer need
    df.drop(['o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_reb', 'o_asts', 'o_to',
            'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_reb', 'd_stl'], axis=1, inplace=True)

    return df


def clean_teams_players(df):
    df = basic_clean(df)

    df['TotalGP'] = df.apply(lambda x: x['GP'] + x['PostGP'], axis=1)
    df['Points'] = df.apply(lambda x: x['PostPoints'] + x['points'], axis=1)
    df['TotalMinutes'] = df.apply(
        lambda x: x['minutes'] + x['PostMinutes'], axis=1)
    df['TotaloRebounds'] = df.apply(
        lambda x: x['oRebounds'] + x['PostoRebounds'], axis=1)
    df['TotaldRebounds'] = df.apply(
        lambda x: x['dRebounds'] + x['PostdRebounds'], axis=1)
    df['TotalRebounds'] = df.apply(
        lambda x: x['rebounds'] + x['PostRebounds'], axis=1)
    df['TotalAssists'] = df.apply(
        lambda x: x['assists'] + x['PostAssists'], axis=1)
    df['TotalSteals'] = df.apply(
        lambda x: x['steals'] + x['PostSteals'], axis=1)
    df['TotalBlocks'] = df.apply(
        lambda x: x['blocks'] + x['PostBlocks'], axis=1)
    df['TotalTurnovers'] = df.apply(
        lambda x: x['turnovers'] + x['PostTurnovers'], axis=1)
    df['TotalPF'] = df.apply(lambda x: x['PF'] + x['PostPF'], axis=1)
    df['TotalfgAttempted'] = df.apply(
        lambda x: x['fgAttempted'] + x['PostfgAttempted'], axis=1)
    df['TotalfgMade'] = df.apply(
        lambda x: x['fgMade'] + x['PostfgMade'], axis=1)
    df['TotalftAttempted'] = df.apply(
        lambda x: x['ftAttempted'] + x['PostftAttempted'], axis=1)
    df['TotalftMade'] = df.apply(
        lambda x: x['ftMade'] + x['PostftMade'], axis=1)
    df['TotalthreeAttempted'] = df.apply(
        lambda x: x['threeAttempted'] + x['PostthreeAttempted'], axis=1)
    df['TotalthreeMade'] = df.apply(
        lambda x: x['threeMade'] + x['PostthreeMade'], axis=1)
    df['TotalGS'] = df.apply(lambda x: x['GS'] + x['PostGS'], axis=1)
    df['TotalDQ'] = df.apply(lambda x: x['dq'] + x['PostDQ'], axis=1)

    df = df.drop(['GP', 'PostGP', 'points', 'PostPoints', 'minutes', 'PostMinutes', 'oRebounds', 'PostoRebounds', 'dRebounds', 'PostdRebounds', 'rebounds', 'PostRebounds', 'assists', 'PostAssists', 'steals', 'PostSteals', 'blocks', 'PostBlocks',
                  'turnovers', 'PostTurnovers', 'PF', 'PostPF', 'fgAttempted', 'PostfgAttempted', 'fgMade', 'PostfgMade', 'ftAttempted', 'PostftAttempted', 'ftMade', 'PostftMade', 'threeAttempted', 'PostthreeAttempted', 'threeMade', 'PostthreeMade', 'GS', 'PostGS', 'dq', 'PostDQ'], axis=1)

    return df
