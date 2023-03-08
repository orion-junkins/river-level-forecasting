from datetime import datetime
import pandas as pd
from statistics import mean


class Evaluator:
    """
    Evaluates the performance of a proudction model over time. Operates on a dataset with a single y_true column and multiple y_pred columns where each y_pred column is the result of a forecast issued at a different time.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Creates a new Evaluator instance.

        Args:
            data (pd.DataFrame): A dataframe with a single y_true column and multiple y_pred columns where each y_pred column is the result of a forecast issued at a different time.
        """
        self.data = self.process_data(data)
        self.y_true = self.data["level_true"]
        self.y_pred = self.data.drop(columns="level_true")

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the data to remove any rows that are missing values.

        Args:
            data (pd.DataFrame): A dataframe with a single y_true column and multiple y_pred columns where each y_pred column is the result of a forecast issued at a different time.

        Returns:
            pd.DataFrame: A dataframe with no missing values in the level_true column and at least one forecasted value in each row.
        """
        data = data.dropna(thresh=2)
        data = data.dropna(subset=["level_true"])
        return data

    @property
    def raw_errors(self) -> pd.DataFrame:
        """
        Calculates the raw errors between the y_true and y_pred values.

        Returns:
            pd.DataFrame: A 2D dataframe with the raw errors between the y_true and y_pred values.
        """
        errors = self.y_pred.sub(self.y_true, axis="index").abs()
        return errors

    @property
    def df_mape(self) -> pd.DataFrame:
        """
        Calculates the mean absolute percentage error for each window size.

        Returns:
            pd.DataFrame: A dataframe with the mean absolute percentage error for each window size.
        """
        return pd.DataFrame.from_dict(self.mape_by_window, orient='index').sort_index()

    @property
    def df_mae(self) -> pd.DataFrame:
        """
        Calculates the mean absolute error for each window size.

        Returns:
            pd.DataFrame: A dataframe with the mean absolute error for each window size.
        """
        return pd.DataFrame.from_dict(self.mae_by_window, orient='index').sort_index()

    @property
    def mape_by_window(self) -> dict[pd.Timedelta, float]:
        """
        Calculates the mean absolute percentage error for each window size.

        Returns:
            dict[pd.Timedelta, float]: A dictionary with the mean absolute percentage error for each window size.
        """
        mape = {}
        for window_size in self.percent_errors_by_window.keys():
            mape[window_size] = mean(self.percent_errors_by_window[window_size])
        return mape

    @property
    def mae_by_window(self) -> dict[pd.Timedelta, float]:
        """
        Calculates the mean absolute error for each window size.

        Returns:
            dict[pd.Timedelta, float]: A dictionary with the mean absolute error for each window size.
        """
        mae = {}
        for window_size in self.absolute_errors_by_window.keys():
            mae[window_size] = mean(self.absolute_errors_by_window[window_size])
        return mae

    @property
    def absolute_errors_by_window(self) -> dict[pd.Timedelta, list[float]]:
        """
        Calculates the absolute errors between the y_true and y_pred values for each window size.

        Returns:
            dict[pd.Timedelta, list[float]]: A dictionary with the errors between the y_true and y_pred values for each window size.
        """
        return self.errors_grouped_by_window(absolute=True)

    @property
    def percent_errors_by_window(self) -> dict[pd.Timedelta, list[float]]:
        """
        Calculates the percentage errors between the y_true and y_pred values for each window size.

        Returns:
            dict[pd.Timedelta, list[float]]: A dictionary with the errors between the y_true and y_pred values for each window size.
        """
        return self.errors_grouped_by_window(absolute=False)

    def errors_grouped_by_window(self, absolute: bool = True) -> dict[pd.Timedelta, list[float]]:
        """
        Calculates the errors between the y_true and y_pred values for each window size.

        Args:
            absolute (bool): Whether to calculate the absolute errors or the percentage errors.

        Returns:
            dict[pd.Timedelta, list[float]]: A dictionary with the errors between the y_true and y_pred values for each window size.
        """
        errors = {}
        for issue_time in self.raw_errors.columns:
            for pred_time in self.raw_errors.index:
                y_true = self.y_true[pred_time]
                y_hat = self.y_pred[issue_time][pred_time]

                if (pd.isna(y_true) or pd.isna(y_hat)):
                    continue

                error = abs(y_true - y_hat)
                if not absolute:
                    error = error / y_true

                window_size = pred_time - datetime.strptime(issue_time, "%y-%m-%d_%H-%M")
                if window_size not in errors:
                    errors[window_size] = [error]
                else:
                    errors[window_size].append(error)
        return errors


def build_evaluator_from_csv(path: str = "data/inference_eval_example.csv") -> Evaluator:
    """
    Factory method to ensure that a csv is read as expected. Use this factory or pass the same kwargs shown below when building elsewhere.

    Builds an Evaluator instance from a csv file. Expects a datetime index, a single y_true column and multiple y_pred columns where each y_pred column is the result of a forecast issued at a different time.

    Args:
        path (str): The path to the csv file.

    Returns:
        Evaluator: An Evaluator instance.
    """
    data = pd.read_csv(path, index_col="datetime", parse_dates=True)

    evaluator = Evaluator(data)
    return evaluator
