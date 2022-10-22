import pytz
import pandas as pd


def format_level_data(df) -> pd.DataFrame:
    """
    Take in a dataframe of level data and handle basic formatting.
        - Convert index to UTC
        - Drop all data excess of hourly frequency
        - Remove duplicates in index
        - Set frequency as hourly
        - Impute NaNs based on adjacent values where possible
        - Drop remaining NaNs

    Args:
        df (pd.DataFrame): Unformatted DataFrame.

    Returns:
        (pd.DataFrame): Formatted DataFrame.
    """
    # Convert index to utc timestamps
    df.index = df.index.map(lambda x: x.astimezone(pytz.utc))

    if df.index[0].minute != 0:
        df.drop([df.index[0]], inplace=True)
    if not (df.index[0].minute == 0):
        raise Exception("Error: failed to coerce index to hourly.")

    # Remove duplicated entries
    df = df.loc[~df.index.duplicated(), :]

    # Set frequency as hourly
    df = df.asfreq('H')

    # Compute forward/back filled data
    for_fill = df.fillna(method='ffill')
    back_fill = df.fillna(method='bfill')
    # For every column in the dataframe,
    for col in df.columns:
        # Average the forward and back filled values
        df[col] = (for_fill[col] + back_fill[col])/2

    # Drop any rows remaining which have NaN values (generally first and/or last rows)
    df.dropna(inplace=True)

    # Confirm imputation worked
    if not (df.isna().sum().sum() == 0):
        raise Exception("Error: Failed to remove all NaNs.")

    return df
