# Merge the dataframes df_players and df_players_teams
def merge_player_info(df_players, df_players_teams):
    df_merged = df_players_teams.merge(
        df_players, left_on='playerID', right_on='bioID')
    df_merged = df_merged.drop(['bioID'], axis=1)
    return df_merged

# Merge the dataframes related to coaches with the current dataframe
def merge_coach_info(df_teams_merged, df_coach_ratings, df_coaches):
    
    # merge df_coaches features year and tmID when stint = 0 to df_coach_ratings and then to df_teams_merged
    df_coaches = df_coaches[df_coaches['stint'] == 0]
    df_coaches = df_coaches[['coachID', 'tmID', 'year']]


    df_coach_ratings = df_coach_ratings.merge(
        df_coaches, left_on=['coachID'],right_on=['coachID'], how="right")


    df_teams_merged = df_teams_merged.merge(
        df_coach_ratings, left_on=['tmID', 'year'], right_on=['tmID', 'year'], how='left')
    
    df_teams_merged = df_teams_merged.fillna(0) 

    return df_teams_merged

# Separate the awards into two dataframes, one for players and one for coaches
def separate_awards_info(df_awards_players, year):
    df_awards_players = df_awards_players[df_awards_players['year'] < year]
    # Get the awards that contain the word coach in any case
    df_awards_coaches = df_awards_players[df_awards_players['award'].str.contains(
        'coach', case=False)]
    df_awards_players = df_awards_players[~df_awards_players['award'].str.contains(
        'coach', case=False)]

    return df_awards_players, df_awards_coaches

# Assign the number of awards to each player
def merge_awards_info(df_players, df_awards_players, year):
    df_awards_players = df_awards_players[df_awards_players['year'] < year]
    df_awards_players = df_awards_players.groupby(['playerID']).agg({
        'award': 'count'
    })
    df_awards_players = df_awards_players.reset_index()
    # Rename playerID or coach to bioID in awards_players
    df_awards_players = df_awards_players.rename(columns={'playerID': 'bioID'})
    df_awards_players = df_awards_players.rename(columns={'coachID': 'bioID'})
    df_players = df_players.rename(columns={'coachID': 'bioID'})
    df_players = df_players.merge(
        df_awards_players, left_on='bioID', right_on='bioID', how='left')
    df_players['award'] = df_players['award'].fillna(0)
    df_players['award'] = df_players['award'].astype(int)
    return df_players


# Assign the number of awards to each team
def merge_add_awards(df_teams_merged, df_players, df_coaches, year):
    df_teams_merged['awards'] = 0
    # for every player in df_teams_merged playerID add the award of that player
    for index, row in df_teams_merged.iterrows():
        playerID = row['playerID']
        for i in playerID:
            df_player = df_players[df_players['bioID'] == i]
            award = df_player['award'].values[0]
            df_teams_merged.at[index, 'awards'] += award

    # standardize the awards
    df_teams_merged['awards'] = df_teams_merged['awards'] / \
        df_teams_merged['awards'].max()
    
    return df_teams_merged