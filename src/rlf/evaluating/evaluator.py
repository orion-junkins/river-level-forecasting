from datetime import datetime
import pandas as pd
from statistics import mean
from typing import Dict, List


class Evaluator:
    """
    Evaluates the performance of a production model over time. Operates on a dataset with a single level_true column and multiple level_pred columns where each level_pred column is the result of a forecast issued at a different time.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Creates a new Evaluator instance.

        Args:
            data (pd.DataFrame): A dataframe with a single level_true column and multiple level_pred columns where each level_pred column is the result of a forecast issued at a different time.
        """
        self.data = self.process_data(data)
        self.level_true = self.data["level_true"]

    def all_level_preds(self) -> pd.DataFrame:
        return self.data.drop(columns="level_true")
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the data to remove any rows that are missing values. The output dataframe is guaranteed to have no missing values in the level_true column and at least one forecasted value in each row.

        Args:
            data (pd.DataFrame): A dataframe with a single level_true column and multiple level_pred columns where each level_pred column is the result of a forecast issued at a different time.

        Returns:
            pd.DataFrame: A dataframe with no missing values in the level_true column and at least one forecasted value in each row.
        """
        data = data.dropna(subset=["level_true"])
        data = data.dropna(thresh=2)  # Given that there is a non NaN value in the level_true column, drop rows that do not have at least one other non NaN value (2 total non NaN values)
        # Convert all column names to date time objects, then to strings in the format "%Y-%m-%d %H:%M:%S%z" and rename the columns accordingly
        new_col_names = []
        for col in data.columns:
            if col == "level_true":
                new_col_names.append(col)
            else:
                new_col_names.append(pd.to_datetime(col, format='%Y-%m-%d %H:%M:%S%z', utc=True).strftime("%Y-%m-%d %H:%M:%S%z"))
        data.columns = new_col_names
        return data

    def df_mape(self) -> pd.DataFrame:
        """
        Calculates the mean absolute percentage error for each window size.

        Returns:
            pd.DataFrame: A dataframe with the mean absolute percentage error for each window size.
        """
        return pd.DataFrame.from_dict(self.mape_by_window(), orient='index').sort_index()

    def df_mae(self) -> pd.DataFrame:
        """
        Calculates the mean absolute error for each window size.

        Returns:
            pd.DataFrame: A dataframe with the mean absolute error for each window size.
        """
        return pd.DataFrame.from_dict(self.mae_by_window(), orient='index').sort_index()

    def mape_by_window(self) -> Dict[pd.Timedelta, float]:
        """
        Calculates the mean absolute percentage error for each window size.

        Returns:
            Dict[pd.Timedelta, float]: A dictionary with the mean absolute percentage error for each window size.
        """
        percent_errors_by_window = self.percent_errors_by_window()
        mape = {}
        for window_size in percent_errors_by_window.keys():
            mape[window_size] = mean(percent_errors_by_window[window_size])
        return mape

    def mae_by_window(self) -> Dict[pd.Timedelta, float]:
        """
        Calculates the mean absolute error for each window size.

        Returns:
            Dict[pd.Timedelta, float]: A dictionary with the mean absolute error for each window size.
        """
        absolute_errors_by_window = self.absolute_errors_by_window()
        mae = {}
        for window_size in absolute_errors_by_window.keys():
            mae[window_size] = mean(absolute_errors_by_window[window_size])
        return mae

    def absolute_errors_by_window(self) -> Dict[pd.Timedelta, List[float]]:
        """
        Calculates the absolute errors between the level_true and level_pred values for each window size.

        Returns:
            Dict[pd.Timedelta, List[float]]: A dictionary with the errors between the level_true and level_pred values for each window size.
        """
        return self.errors_grouped_by_window(absolute=True)

    def percent_errors_by_window(self) -> Dict[pd.Timedelta, List[float]]:
        """
        Calculates the percentage errors between the level_true and level_pred values for each window size.

        Returns:
            Dict[pd.Timedelta, List[float]]: A dictionary with the errors between the level_true and level_pred values for each window size.
        """
        return self.errors_grouped_by_window(absolute=False)

    def errors_grouped_by_window(self, absolute: bool = True) -> Dict[pd.Timedelta, List[float]]:
        """
        Calculates the errors between the level_true and level_pred values for each window size.

        Args:
            absolute (bool): Whether to calculate the absolute errors or the percentage errors.

        Returns:
            Dict[pd.Timedelta, List[float]]: A dictionary with the errors between the level_true and level_pred values for each window size.

        Raises:
            ZeroDivisionError: If the level_true value is 0 and absolute is False. This is because the percentage error cannot be calculated when the level_true value is 0.
        """
        errors = {}
        all_level_preds = self.all_level_preds()
        for issue_time in all_level_preds.columns:
            for pred_time in all_level_preds.index:
                level_true = self.level_true[pred_time]
                level_pred = all_level_preds[issue_time][pred_time]

                if (pd.isna(level_true) or pd.isna(level_pred)):
                    continue

                error = abs(level_true - level_pred)
                if not absolute:
                    if level_true == 0:
                        raise (ZeroDivisionError("Cannot calculate percentage error when level_true is 0"))
                    error = error / level_true
                # try:
                issue_dt = datetime.strptime(issue_time, "%Y-%m-%d %H:%M:%S%z")
                naive_issue_dt = issue_dt.replace(tzinfo=None)
                naive_pred_time = pred_time.replace(tzinfo=None)
                window_size = naive_pred_time - naive_issue_dt
                # except ValueError:
                #     tz_time = datetime.strptime(issue_time, "%y-%m-%d_%H-%M")
                #     naive_time = tz_time.replace(tzinfo=None)
                #     window_size = pred_time - naive_time

                if window_size.seconds // 60 % 60 != 0:
                    continue
                if window_size not in errors:
                    errors[window_size] = [error]
                else:
                    errors[window_size].append(error)
        return errors

    def filter_timestamps(self, other_evaluator):
        """Drop any columns in self.data that are not in other_evaluator.data. Convert the column names to datetime objects before comparing.

        Args:
            other_evaluator (Evaluator): The other evaluator to compare against.
        """
        self.data = self.data[self.data.columns.intersection(other_evaluator.data.columns)]


def build_evaluator_from_csv(path: str = "data/inference_eval_example.csv") -> Evaluator:
    """
    Factory method to ensure that a csv is read as expected. Use this factory or pass the same kwargs shown below when building elsewhere.

    Builds an Evaluator instance from a csv file. Expects a datetime index, a single level_true column and multiple level_pred columns where each level_pred column is the result of a forecast issued at a different time.

    Args:
        path (str): The path to the csv file.

    Returns:
        Evaluator: An Evaluator instance.
    """
    data = pd.read_csv(path, index_col="datetime", parse_dates=True)

    evaluator = Evaluator(data)
    return evaluator



# %%
