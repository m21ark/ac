import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from funcs.statistical_analysis import *
from funcs.results_analysis import *
from funcs.merge import *
from funcs.clean import *

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


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


def expanding_window_decay_cross_validation(data, model_func, features, year, decay_rate=0.5):
    clf = model_func()

    for i in range(1, year):
        train_data = data[data['year'] == i]

        X_train = train_data[features]
        y_train = train_data['playoff']

        # Apply weight to older data
        weight = decay_rate ** (year - i - 1)
        sample_weight = [weight] * len(X_train)

        if type(model_func()).__name__ in ["KNeighborsClassifier", "MLPClassifier"]:
            # This model don't support sample weights
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, sample_weight=sample_weight)

    return clf


def global_merge(df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players, year):

    df_awards_players, df_awards_coaches = separate_awards_info(
        df_awards_players, year)

    df_merged = merge_player_info(df_players, df_players_teams)

    # collumn tmID and stint should be dropped
    df_players_teams = df_players_teams.drop(['tmID', 'stint'], axis=1)

    df_defensive_player_stats = defensive_player_ranking(df_players_teams, year=year-1)
    df_offensive_player_stats = player_offensive_rating(df_players_teams, year=year-1)
    df_player_ratings = player_rankings(df_merged, year=year-1)
    df_players_teams = player_in_team_by_year(df_merged)
    df_players_teams = team_mean(df_players_teams, df_player_ratings, df_offensive_player_stats, df_defensive_player_stats)

    # use add_player_stats and add all information to df_players_teams
    #df_players_stats = add_player_stats(df_merged, year=year-1)

    # print(df_players_stats)
    # df_players_teams join with df_players_stats and group by tmID and year
    # df_players_teams = df_players_teams.merge( df_players_stats, on=['tmID', 'year'], how='left')

    #df_players_teams = df_players_teams.drop(['playerID_x'], axis=1)
    # group by tmID and year 
    #df_players_teams = df_players_teams.groupby(['tmID', 'year']).agg(
    #     {
    #     'Points': 'mean',
    #     'TotaloRebounds': 'mean',
    #     'TotaldRebounds': 'mean',
    #     'TotalAssists': 'mean',
    #     'TotalSteals': 'mean',
    #     'TotalBlocks': 'mean',
    #     'TotalTurnovers': 'mean',
    #     'TotalPF': 'mean',
    #     'TotalfgMade': 'mean',
    #     'TotalftMade': 'mean',
    #     'TotalthreeMade': 'mean',
    # }
    # ).reset_index()

    # print(df_players_teams) 
    

    
    # print(df_teams)
    #df_teams = team_rankings(df_teams, year)

    df_teams_merged = df_players_teams.merge(
        df_teams[['tmID', 'year', 'confID', 'playoff']], on=['tmID', 'year'], how='left')


    # df_merged = merge_coach_info(df_teams_merged, df_coaches)
    df_coach_ratings = coach_ranking(df_coaches, year=year)


    # df_coaches = df_coaches.rename(columns={'bioID': 'coachID'})
    # print (df_teams_merged)

    df_teams_merged = merge_coach_info(
        df_teams_merged, df_coach_ratings, df_coaches)
    
    

    df_players = merge_awards_info(df_players, df_awards_players, year)
    df_coaches = merge_awards_info(df_coaches, df_awards_coaches, year)

    df_teams_merged = merge_add_awards( 
        df_teams_merged, df_players, df_coaches, year)

    df_teams_merged = df_teams_merged.drop(['coachID'], axis=1)
    df_teams_merged = df_teams_merged.drop(['playerID'], axis=1)

    
    # df_teams_merged = df_teams_merged.drop(['awards'], axis=1)

    return df_teams_merged


def model_classification(df_teams_merged, year, model = lambda: RandomForestClassifier(n_estimators=100, random_state=42), grid=False, parameters={}, lightGBM=False):
    # teams on year

    test = df_teams_merged[df_teams_merged['year'] == year]
    train = df_teams_merged[df_teams_merged['year'] < year]

    # use a MLP to predict the playoff entry

    # convert the confID to a number
    train['confID'] = train['confID'].replace(['EA', 'WE'], [0, 1])
    test['confID'] = test['confID'].replace(['EA', 'WE'], [0, 1])

    if grid:
        grid_search = GridSearchCV(model(), parameters, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(train.drop(['playoff', 'year', 'tmID'], axis=1), train['playoff'])  # Replace X and y with your data

        print(f"Best parameters for {grid_search.best_estimator_.__class__.__name__}: {grid_search.best_params_}")
        print(f"Best score for {grid_search.best_estimator_.__class__.__name__}: {grid_search.best_score_}")
        model = lambda: grid_search.best_estimator_

    predictions = []
    
    if not lightGBM:

        ## normalize the data
        #scaler = StandardScaler()
        #scaler.fit(train.drop(['playoff', 'year', 'tmID'], axis=1))
        #train = scaler.transform(train.drop(['playoff', 'year', 'tmID'], axis=1))
        #test = scaler.transform(test.drop(['playoff', 'year', 'tmID'], axis=1))

        clf = expanding_window_decay_cross_validation(
            train.drop(['tmID'], axis=1), model, train.drop(['playoff', 'year', 'tmID'], axis=1).columns, year)
        #clf = best_model
    
        clf.fit(train.drop(['playoff', 'year', 'tmID'], axis=1), train['playoff'])
        predictions = clf.predict_proba(test.drop(['playoff', 'year', 'tmID'], axis=1))[:, 1]
    else:
        test = lgb.Dataset(test.drop(['playoff', 'year', 'tmID'], axis=1), label=test['playoff'])
        train = lgb.Dataset(train.drop(['playoff', 'year', 'tmID'], axis=1), label=train['playoff'])
        params = {
        'objective': 'binary',  # 'binary' for binary classification
        'metric': 'binary_error',  # Evaluation metric
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        }

        num_round = 100  # Number of boosting rounds
        model = lgb.train(params, train, num_round, valid_sets=[test], early_stopping_rounds=10)
        predictions = model.predict_proba(test.drop(['playoff', 'year', 'tmID'], axis=1))

    test['predictions'] = predictions
    df_teams_merged['predictions'] = 0
    df_teams_merged.loc[df_teams_merged['year'] == year, 'predictions'] = predictions


    return df_teams_merged, clf


def pipeline_clf(year = 10):
    if year > 11 or year < 1:
        raise ValueError("Year must be between 2 and 11")

    # Load the clean datasets
    df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players = load_data()

    df_teams_merged = global_merge(df_teams, df_teams_post, df_series_post,
                                   df_players, df_players_teams, df_coaches, df_awards_players, year)

    return df_teams_merged

def pipeline_year(year=10, model =lambda: RandomForestClassifier(n_estimators=100, random_state=42),  display_results=False, lightGBM=False):

    if year > 11 or year < 2:
        raise ValueError("Year must be between 2 and 11")

    # Load the clean datasets
    df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players = load_data()


    df_teams_merged = global_merge(df_teams, df_teams_post, df_series_post,
                                   df_players, df_players_teams, df_coaches, df_awards_players, year)

    # if (lightGBM):
    df_teams_merged, clf = model_classification(df_teams_merged, year, model=model)
    # else:
    #     train = df_teams_merged[df_teams_merged['year'] < year]
    #     test = df_teams_post[df_teams_post['year'] == year]
    #     train = lgb.Dataset(train.drop(['playoff', 'year', 'tmID'], axis=1), label=train['playoff'])
    #     params = {
    #     'objective': 'binary',  # 'binary' for binary classification
    #     'metric': 'binary_error',  # Evaluation metric
    #     'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     }



    # add the coach ratings to the df_teams_merged on the coachID and yeat
    # df_teams_merged = df_teams_merged.merge(
    #   df_coach_ratings, on=['coachID'], how='left')

    # print year 11
    # print(df_teams_merged[df_teams_merged['year'] == 11])


    df_teams, ea_teams, we_teams = classify_playoff_entry(
        df_teams_merged, year)

    ea_predictions = ea_teams['tmID'].unique()
    we_predictions = we_teams['tmID'].unique()
    

    total_precision = calculate_playoff_accuracy(
        year, ea_predictions, we_predictions, display_results)

    return total_precision

def pipeline_year_grid_search(year=10, model = lambda: RandomForestClassifier(), parameters={}, display_results=False):

    if year > 11 or year < 2:
        raise ValueError("Year must be between 2 and 11")
    
    # Load the clean datasets
    df_teams, df_teams_post, df_series_post, df_players, df_players_teams, df_coaches, df_awards_players = load_data()


    df_teams_merged = global_merge(df_teams, df_teams_post, df_series_post,
                                   df_players, df_players_teams, df_coaches, df_awards_players, year)

    
    df_teams_merged, clf = model_classification(df_teams_merged, year, model=model, grid=True, parameters=parameters)

    df_teams, ea_teams, we_teams = classify_playoff_entry(df_teams_merged, year)

    ea_predictions = ea_teams['tmID'].unique()
    we_predictions = we_teams['tmID'].unique()

    total_precision = calculate_playoff_accuracy(year, ea_predictions, we_predictions, display_results)

    return total_precision



def check_accuracy_by_year():
    accs = []
    years = list(range(2, 11))

    for year in years:
        acc = pipeline_year(year)
        accs.append(acc)

    # plot the accuracy line graph
    plt.plot(years, accs, label="Accuracy", marker='o', linestyle='-')

    # add labels for each data point
    # for i, acc in enumerate(accs):
      #  plt.text(years[i], acc, f"{acc:.2f}", ha="center", va="bottom")

    # add legend
    plt.legend()

    # set Y-axis limits
    plt.ylim(50, 100)

    plt.xlabel("Year")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by year")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    apply_cleaning()
    # check_accuracy_by_year()
    print(pipeline_year(10))
