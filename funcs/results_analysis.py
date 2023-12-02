import itertools
from colorama import Fore, Style
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Calculate the accuracy of the predictions
def calculate_playoff_accuracy(year, predicted_ea_playoffs, predicted_we_playoffs, display_results=True):

    # Auxiliary function to color the text
    def color_text(predicted, actual):
        colored_text = []
        for team in predicted:
            if team in actual:
                colored_text.append(Fore.GREEN + team + Style.RESET_ALL)
            else:
                colored_text.append(Fore.RED + team + Style.RESET_ALL)

        return ', '.join(colored_text)

    results_df = pandas.read_csv("dataset/cleaned/results.csv")

    year_df_we = results_df[(results_df['year'] == year)
                            & (results_df['confID'] == 'WE')]
    year_df_ea = results_df[(results_df['year'] == year)
                            & (results_df['confID'] == 'EA')]

    actual_we_playoffs = year_df_we[year_df_we['playoff']
                                    == 1]['tmID'].tolist()
    actual_ea_playoffs = year_df_ea[year_df_ea['playoff']
                                    == 1]['tmID'].tolist()

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

    sorted_team_list = list(predicted_ea_playoffs) + \
        list(predicted_we_playoffs)

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
    we_incorrect_count = len(we_incorrect)
    ea_incorrect_count = len(ea_incorrect)
    all_we_count = len(all_we_teams)
    all_ea_count = len(all_ea_teams)

    TP = we_correct_count + ea_correct_count
    FP = we_incorrect_count + ea_incorrect_count
    FN = FP
    TN = (all_we_count + all_ea_count) - TP - FN - FP 

    total_accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    total_precision = (TP / (TP + FP)) * 100
    total_recall = (TP / (TP + FN)) * 100
    total_f1 = (2 * total_precision * total_recall) / (total_precision + total_recall)

    if display_results:
        print("\n")
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
        print(f"{'Total precision:':<20}{total_precision:.2f}%")
        print(f"{'Total recall:':<20}{total_recall:.2f}%")
        print(f"{'Total f1:':<20}{total_f1:.2f}%\n")

        # display confusion matrix
        cm = np.array([[TP, FP], [FN, TN]])
        display_confusionMatrix(cm)

    return total_precision

# Display the confusion matrix
def display_confusionMatrix(cm):
    classes = ['Playoff', 'Eliminated']
    plt.matshow(cm)
    plt.suptitle('Confusion matrix')
    total = sum(sum(cm))
    plt.title('Total cases: {}'.format(total))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            perc = round(cm[i, j] / total * 100, 1)
            plt.text(j, i, f"{format(cm[i, j], '.0f')} : {perc}%", horizontalalignment="center",
                     color="black" if cm[i, j] > cm.max() / 2 else "white")

    plt.show()

# 
def remove_currently_unknown_data(df_teams, year):
    # Remove all columns that are unknown at the start of the season on a given year
    # And replace them by the mean of the previous 2 years

    columns_with_unknown_data = [
        'o_oreb', 'o_dreb', 'o_pf', 'o_stl', 'o_blk', 'o_pts',
        'd_oreb', 'd_dreb', 'd_asts', 'd_pf', 'd_to', 'd_blk', 'd_pts',
        'min', 'attend', 'RoundReached', 'winPercentage',
        'homeWinPercentage', 'awayWinPercentage', 'of_goal',
        'of_3pt', 'of_throw', 'of_reb', 'of_assist', 'df_goal',
        'df_3pt', 'df_throw', 'df_reb', 'df_steal'
    ]

    if year <= 2:
        # For the 1st and 2nd years, we don't have data from 2 previous years
        # So we just remove the columns
        df_teams = df_teams.drop(columns=columns_with_unknown_data)
        return df_teams

    for column in columns_with_unknown_data:
        previous_years = [year - 1, year - 2]

        # Calculate the mean of the previous 2 years
        mean_values = df_teams[df_teams['year'].isin(
            previous_years)][column].mean()

        # Replace missing values in the current year with the mean
        df_teams.loc[df_teams['year'] == year,
                     column] = df_teams.loc[df_teams['year'] == year, column].fillna(mean_values)

    # Set "playoff" to be blank for the current year
    df_teams.loc[df_teams['year'] == year, 'playoff'] = None

    return df_teams
