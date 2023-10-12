import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style


# =============================== UTILS ===============================

def remove_redundant_cols(df):
    return df.loc[:, df.nunique() > 1]


def replace_col_with_ids(df, col_name):
    unique_awards = df[col_name].unique()
    award_mapping = {award: i for i,
                     award in enumerate(unique_awards, start=1)}

    df[col_name] = df[col_name].replace(award_mapping)
    return df


def basic_clean(df, cols=[]):
    df = df.loc[:, df.nunique() > 1]
    df = df.drop(columns=cols, axis=1)
    df = remove_redundant_cols(df)
    return df


# =============================== CLEANING ===============================

def clean_players(df):
    df = basic_clean(df, ['college', 'collegeOther', 'birthDate', 'deathDate'])
    return df


def clean_awards_players(df):
    df = basic_clean(df, ['award'])
    return df


def clean_coaches(df):
    df = basic_clean(df)
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
