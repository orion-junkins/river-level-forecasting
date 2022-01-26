import pandas as pd
from IPython.display import display

def merge(df_list, verbose=False):
    temp_df_list = df_list.copy()
    df_merged = temp_df_list.pop(0)

    for df in temp_df_list:
        display(df)
        df_merged = pd.merge(df_merged, df, on="datetime")

    if verbose:
        print("Data merged. Full data frame following merge:")
        display(df_merged)
    
    return df_merged
