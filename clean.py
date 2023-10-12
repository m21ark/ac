import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style



def pipeline_year(year = 10):
    # Load the datasets
    df_teams = pandas.read_csv("dataset/teams.csv")
    df_teams_post = pandas.read_csv("dataset/teams_post.csv")
    df_series_post = pandas.read_csv("dataset/series_post.csv")
    df_players = pandas.read_csv("dataset/players.csv")
    df_players_teams = pandas.read_csv("dataset/players_teams.csv")
    df_coaches = pandas.read_csv("dataset/coaches.csv")
    df_awards_players = pandas.read_csv("dataset/awards_players.csv")

    dfs = [df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players]
    dfs_names = ["teams", "teams_post", "series_post", "players", "players_teams", "coaches", "awards_players"]

    df_players_teams = clean_teams_players(df_players_teams)
    df_players = clean_players(df_players)

    df_merged = merge_player_info(df_players, df_players_teams)

    # collumn tmID and stint should be dropped
    df_players_teams = df_players_teams.drop(['tmID', 'stint'], axis=1)

    df_player_ratings = player_rankings(df_merged, year=year-1)

    df_players_teams = player_in_team_by_year(df_merged)

    df_players_teams = team_mean(df_players_teams, df_player_ratings)

    # call model
    #print(df_players_teams, df_teams)
    # call classification method
    df_teams, ea_teams, we_teams = classify_playoff_entry(df_players_teams, df_teams, year)

    ea_predictions = ea_teams['tmID'].unique()
    we_predictions = we_teams['tmID'].unique()

    calculate_playoff_accuracy(year, ea_predictions, we_predictions)

    return df_teams

def classify_playoff_entry(df_players_teams, df_teams, year):
    df_teams_at_year = df_players_teams[df_players_teams.year == year]
    
    # add to the df_teams at year the division from df_teams
    df_teams_at_year = df_teams_at_year.merge(df_teams[['tmID', 'year', 'confID']], on=['tmID', 'year'], how='left')

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
        row['playoff'] = 'Y'

    for index, row in we_playoff_teams.iterrows():
        row['playoff'] = 'Y'

    return df_teams, ea_playoff_teams, we_playoff_teams


def color_text(predicted, actual):
    colored_text = []
    for team in predicted:
        if team in actual:
            colored_text.append(Fore.GREEN + team + Style.RESET_ALL)
        else:
            colored_text.append(Fore.RED + team + Style.RESET_ALL)

    return ', '.join(colored_text)


def calculate_playoff_accuracy(year, predicted_ea_playoffs, predicted_we_playoffs):

    results_df = pandas.read_csv("dataset/results.csv")

    year_df_we = results_df[(results_df['year'] == year)
                            & (results_df['confID'] == 'WE')]
    year_df_ea = results_df[(results_df['year'] == year)
                            & (results_df['confID'] == 'EA')]

    actual_we_playoffs = year_df_we[year_df_we['playoff']
                                    == 'Y']['tmID'].tolist()
    actual_ea_playoffs = year_df_ea[year_df_ea['playoff']
                                    == 'Y']['tmID'].tolist()

    all_we_teams = year_df_we['tmID'].tolist()
    all_ea_teams = year_df_ea['tmID'].tolist()

    team_conf_dict = {}

    for team in all_we_teams:
        team_conf_dict[team] = 'WE'
    for team in all_ea_teams:
        team_conf_dict[team] = 'EA'

    we_incorrect = []
    ea_incorrect = []
    we_correct = []
    ea_correct = []

    sorted_team_list = list(predicted_ea_playoffs) + list(predicted_we_playoffs)

    for team in sorted_team_list:
        if team_conf_dict[team] == 'WE':
            if team in actual_we_playoffs:
                we_correct.append(team)
            else:
                we_incorrect.append(team)
        else:
            if team in actual_ea_playoffs:
                ea_correct.append(team)
            else:
                ea_incorrect.append(team)

    we_correct_count = len(we_correct)
    ea_correct_count = len(ea_correct)

    total_accuracy = ((we_correct_count + ea_correct_count) / 8.0) * 100

    print("=" * 40)
    print(f"{'Year:':<20}{year}")
    print("=" * 40)
    print("WE")
    print(f"{'Guesses:':<15}{color_text(we_correct, actual_we_playoffs) + ', ' + color_text(we_incorrect, actual_we_playoffs) }")
    print(f"{'Missed:':<15}{', '.join(set(actual_we_playoffs) - set(we_correct))}")
    print("=" * 40)
    print("EA")
    print(f"{'Guesses:':<15}{color_text(ea_correct, actual_ea_playoffs) + ', ' + color_text(ea_incorrect, actual_ea_playoffs)}")
    print(f"{'Missed:':<15}{', '.join(set(actual_ea_playoffs) - set(ea_correct))}")
    print("=" * 40)
    print(f"{'Total accuracy:':<20}{total_accuracy:.2f}%")


def merge_player_info(df_players, df_players_teams):
    df_merged = df_players_teams.merge(df_players, left_on='playerID', right_on='bioID')

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
        mean_l.append(top_12_players ['predictions'].mean())


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

    df_merged = df_merged.drop(['TotalMinutes','TotalGP'], axis=1)

    players_id = df_merged['playerID']
    year = df_merged['year']

    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # drop all non numerical atributes
    df_merged = df_merged.select_dtypes(include=['float64', 'int64'])
    

    df_merged = df_merged.drop(['year'], axis=1)

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
    df_merged['year'] = year

    # drop all atributes exept playerID, medium and predictions
    df_pred = df_merged[['year','playerID', 'medium', 'predictions']]


    return df_pred


def player_rankings(df_merged, year = 10):

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
    ## Existe um problema com esta abordagem, jogadores que jogaram pouco mas bem podem ter uma vantagem ... TODO: testar com isto
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

    

df_coaches = pandas.read_csv("dataset/coaches.csv")



print(df_coaches)

#df = pipeline_year(10)

