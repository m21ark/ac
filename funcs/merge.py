
def merge_player_info(df_players, df_players_teams):
    df_merged = df_players_teams.merge(
        df_players, left_on='playerID', right_on='bioID')
    df_merged = df_merged.drop(['bioID'], axis=1)
    return df_merged


def merge_coach_info(df_teams, df_coaches):
    df_coaches = df_coaches.rename(columns={'bioID': 'coachID'})
    df_coaches = df_coaches[df_coaches['stint'] <= 1]
    df_coaches = df_coaches.drop(['stint'], axis=1)
    df_merged = df_teams.merge(
        df_coaches, left_on=['tmID', 'year'], right_on=['tmID', 'year'])

    return df_merged


def separate_awards_info(df_awards_players, year):
    df_awards_players = df_awards_players[df_awards_players['year'] < year]
    # Get the awards that contain the word coach in any case
    df_awards_coaches = df_awards_players[df_awards_players['award'].str.contains(
        'coach', case=False)]
    df_awards_players = df_awards_players[~df_awards_players['award'].str.contains(
        'coach', case=False)]

    return df_awards_players, df_awards_coaches


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
