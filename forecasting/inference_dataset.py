from general_utilities.dataset_utilities import pre_process

class InferenceDataset:
    """
    Dataset abstraction that fetches, processes and exposes needed X and y datasets for inference given a CatchmentData instance.
    """
    def __init__(self, catchment_data, scaler, target_scaler) -> None:
        """
        Generate an inference Dataset from a CatchmentData instance using the given test and validation sizes.

        Note that variable plurality indicates the presence of multiple datasets. Ie Xs_train is a list of multiple X_train sets.
        """
        self.catchment_data = catchment_data
        self.scaler = scaler
        self.target_scaler = target_scaler

        self.Xs, self.y = self._get_data()

        # TODO add validation call - ie all X sets are same size, match y sets.
    

    def _get_data(self, update=False):
        # Update if specified 
        if update:
            self.catchment_data.update_for_inference()

        # Fetch
        current_weather, recent_level = self.catchment_data.all_current

        # Process
        Xs_current, y_current = pre_process(current_weather, recent_level, allow_future_X=True)

        # Scale
        Xs_current = self.scaler.transform(Xs_current)
        y_current = self.target_scaler.transform(y_current)

        return(Xs_current, y_current)


    def update(self):
        """
        Update the underlying catchment data for inference with up to date data. This triggers new weather data queries and a rebuild of all current datasets.
        """
        self.Xs_current, self.y_current = self._get_data(update=True)
