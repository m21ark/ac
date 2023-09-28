import pandas


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
