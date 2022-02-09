import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
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


def add_lag(df, timesteps=1, col_to_lag='level', lagged_col_name='level_t_sub_1'):
    df[lagged_col_name] = np.nan
    timesteps = df.shape[0]
    for t in range(1, timesteps):
        df[lagged_col_name][t] = df[col_to_lag][t-1]
    df = df.iloc[1: , :]
    return df

def split_X_y(df, target_col_name='level'):
    y = DataFrame(df[target_col_name])
    df.drop(columns=[target_col_name], inplace=True)
    X = df
    return (X,y)

def scale(df, scaler, fit_scalers=True) -> DataFrame:
    if fit_scalers:
        df[df.columns] = scaler.fit_transform(df[df.columns])
    else:
        df[df.columns] = scaler.transform(df[df.columns])
    return df


def get_all_windows(X, y, window_length=5):
    num_samples = len(X) - window_length
    X = X.values
    y = y.values
    windowed_X = []
    windowed_y = []
    for index in range(num_samples):
        current_window_end = index + window_length
        cur_X_seq = X[index:current_window_end, :]
        windowed_X.append(cur_X_seq)

        windowed_y.append(y[current_window_end])

    X = np.array(windowed_X)
    y = np.array(windowed_y)
    
    return (X, y)

def partition(X, y, window_length=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
    print(len(X_train))
    X_train = X_train[:-window_length]
    print(len(X_train))
    print(len(y_train))
    y_train = y_train[:-window_length]
    print(len(y_train))
    return (X_train, X_test, y_train, y_test)