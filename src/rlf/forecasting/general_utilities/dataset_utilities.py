from darts import timeseries


def pre_process(Xs, y, rolling_sum_columns=[], rolling_mean_columns=[], window_sizes=[10*24, 30*24], allow_future_X=False):
    """
    Pre process data. This includes adding engineered features and trimming datasets to ensure X, y consistency.

    Args:
        Xs (list of dataframes): List of all X sets.
        y (dataframe): Dataframe containing y set.
        allow_future_X (bool, optional): Determines if end date of X sets can exceed end date of y set. Only True for current data which includes forecasts (level data into future is not yet known, but weather is). Defaults to False.

    Returns:
        tuple of (list of TimeSeries, TimeSeries): Tuple containing (processed_Xs, y)
    """
    y = timeseries.TimeSeries.from_dataframe(y)

    processed_Xs = []

    for X_cur in Xs:
        X_cur = add_engineered_features(X_cur)
        X_cur = timeseries.TimeSeries.from_dataframe(X_cur)

        if X_cur.start_time() < y.start_time():
            _, X_cur = X_cur.split_before(y.start_time())    # X starts before y, drop X before y start

        if not allow_future_X:
            if X_cur.end_time() > y.end_time():
                X_cur, _ = X_cur.split_after(y.end_time())  # X ends after y, drop X after y end

        processed_Xs.append(X_cur)

    if y.start_time() < processed_Xs[0].start_time():
        _, y = y.split_before(processed_Xs[0].start_time())  # y starts before X, drop y before X start

    if y.end_time() > processed_Xs[0].end_time():  # y ends after X, drop y after X end
        y, _ = y.split_after(processed_Xs[0].end_time())

    return (processed_Xs, y)


def add_engineered_features(df, rolling_sum_columns=[], rolling_mean_columns=[], window_sizes=[10*24, 30*24]):
    """
    Generate and add engineered features.

    Args:
        df (dataframe): Data from which features should be engineered.

    Returns:
        dataframe: Data including new features.
    """
    df['day_of_year'] = df.index.day_of_year

    for window_size in window_sizes:
        for rolling_sum_col in rolling_sum_columns:
            new_col_name = rolling_sum_col + "_sum_" + window_size
            df[new_col_name] = df[rolling_sum_col].rolling(window=window_size).sum()

        for rolling_mean_col in rolling_mean_columns:
            new_col_name = rolling_mean_col + "_mean_" + window_size
            df[new_col_name] = df[rolling_mean_col].rolling(window=window_size).mean()

    df.dropna(inplace=True)
    return df
