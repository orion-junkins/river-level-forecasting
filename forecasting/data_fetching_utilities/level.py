#%%
import dataretrieval.nwis as nwis
import pandas as pd

def get_historical_level(gauge_id, start="1900-01-01", end=None, parameterCd='00060', drop_cols=["00060_cd", "site_no"], rename_dict={"00060":"level"} ):
    """
    Fetch level data for the given gauge ID. Fetches instant values from start to end.
    Drops and renames columns according to given args.

    Args:
        gauge_id (string): USGS Gauge ID
        start (str, optional): Start date in the form "yyyy-mm-dd". Defaults to "1900-01-01", giving data from start of collection.
        end  (str, optional): End date in the form "yyyy-mm-dd". Defaults to None, giving data til end of collection.
        parameterCd (str, optional): Which parameter to fetch data for. Defaults to '00060' indicated mean level.
        drop_cols (list, optional): Column names to drop if they are present. Defaults to ["00060_cd", "site_no"] (useless metadata).
        rename_dict (dict, optional): Dictionary of default:new defining column renamings. Defaults to {"00060":"level"}.

    Returns:
        df (Pandas dataframe): Formatted dataframe of fetched data
    """
    # Fetch level data
    df = nwis.get_record(sites=gauge_id, service='iv', start=start, end=end, parameterCd=parameterCd)

    # Filter out any columns that are present in the drop_cols list
    drop_cols = list(filter(lambda x: x in df.columns, drop_cols))
    df.drop(columns=drop_cols, inplace=True)

    # Rename columns as specified
    df.rename(columns=rename_dict, inplace=True)

    # Convert index to datetime objects with no timezone
    df.index = pd.to_datetime(df.index)
    df.index = df.index.map(lambda x: x.replace(tzinfo=None))
    df.index = pd.DatetimeIndex(df.index)

    # Force frequency to hourly and fill in NaNs
    df = handle_missing_data(df)

    return df


def handle_missing_data(df):
    if df.index[0].minute != 0:
        df.drop([df.index[0]], inplace=True)
    assert(df.index[0].minute == 0)

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
    assert(df.isna().sum().sum() == 0)

    return df
# %%
