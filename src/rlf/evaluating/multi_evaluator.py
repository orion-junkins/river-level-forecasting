from typing import List

import pandas as pd


class MultiEvaluator:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def joint_df_mape(self, only_mutuals: bool = True) -> List[pd.DataFrame]:
        """Call df_mape on each evaluator and return the results in a list. If only_mutuals is True, only return the results for the mutual window sizes (columns).

        Args:
            only_mutuals (bool, optional): If True, only return the results for the mutual window sizes (columns). Defaults to True.

        Returns:
            List[pd.DataFrame]: _description_
        """
        dfs_mape = []
        for evaluator in self.evaluators:
            df_mape = evaluator.df_mape(only_mutuals=only_mutuals)
            dfs_mape.append(df_mape)

        # Identify which column names all the dataframes have in common.
        common_column_names = dfs_mape[0].columns
        for df_mape in dfs_mape:
            common_column_names = common_column_names.intersection(df_mape.columns)

        # Filter the dataframes to only include the common column names.
        dfs_mape = [df_mape[common_column_names] for df_mape in dfs_mape]
        