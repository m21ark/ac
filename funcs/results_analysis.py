from colorama import Fore, Style
import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def calculate_playoff_accuracy(year, predicted_ea_playoffs, predicted_we_playoffs):

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
