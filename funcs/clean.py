
import pandas as pd


# ======================== Utility functions ========================

def remove_redundant_cols(df):
    return df.loc[:, df.nunique() > 1]


def replace_col_with_ids(df, col_name):
    unique_awards = df[col_name].unique()
    award_mapping = {award: i for i,
                     award in enumerate(unique_awards, start=1)}

    # df[col_name] = df[col_name].replace(award_mapping)
    df.loc[:, col_name] = df.loc[:, col_name].replace(award_mapping)
    return df


def basic_clean(df, cols=[]):
    df = df.loc[:, df.nunique() > 1]
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
    # TODO: Clean series_post
    return df


def clean_teams_post(df):
    df = basic_clean(df)
    # TODO: Clean teams_post
    return df


def clean_teams(df):
    df = basic_clean(df)

    # TODO: check if this is cleaning is correct

    df = df.drop(columns=['franchID',
                 'rank', 'name', "arena", "GP"], axis=1)
    df['confID'] = df['confID'].replace(
        {'West': 0, 'East': 1})  # ver se isto não dá problemas

    # replace Playoffs with 1 and 0 instead of Y and N
    df['playoff'] = df['playoff'].replace('Y', 1)
    df['playoff'] = df['playoff'].replace('N', 0)

    # replace df features firstRound, semis, final with a single feature named RoundReached with values 0, 1, 2, 3
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

    df['RoundReached'] = df['firstRound'] + \
        df['semis'] + df['finals']

    df = df.drop(columns=['firstRound', 'semis', 'finals'], axis=1)

    # add a new feature that indicates the pecentage of games won
    df['winPercentage'] = round((df['homeW'] + df['awayW']) / (
        df['homeW'] + df['awayW'] + df['awayL'] + df['homeL']), 3)
    #
    #
    # Talvez dei bias ao adicionar esta feature ... porém parece que as vitorias em casa são indicadoras de sucesso
    #
    # add a new feature that indicates the pecentage of home games won
    df['homeWinPercentage'] = round(df['homeW'] /
                                    (df['homeW'] + df['homeL']), 3)
    # add a new feature that indicates the pecentage of away games won
    df['awayWinPercentage'] = round(df['awayW'] /
                                    (df['awayW'] + df['awayL']), 3)

    # drop the W and L features
    df = df.drop(
        columns=['homeW', 'homeL', 'awayW', 'awayL', 'won', 'lost'], axis=1)

    # Conf League wins and losses n parece influenciar o acesso a playoffs mas convém ver se tem algo a ver
    # remove confW and confL
    df = df.drop(columns=['confW', 'confL'], axis=1)

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
