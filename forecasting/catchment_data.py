class CatchmentData:
    def __init__(self, catchment_name, weather_provider, level_provider, num_recent_samples=40*24) -> None:
        self.name = catchment_name
        self.weather_provider = weather_provider
        self.level_provider = level_provider
        self.num_recent_samples = num_recent_samples

        self._all_current = (None, None)
        self._all_historical = (None, None)

    @property
    def num_data_sets(self):
        current_weather = self.all_current[0]
        historical_weather = self.all_historical[0]
        assert (len(current_weather) == len(historical_weather))
        return len(current_weather)

    @property
    def all_current(self):
        if self._all_current == (None, None):
            self._get_all_current()

        return self._all_current

    def _get_all_current(self):
        current_weather = self.weather_provider.fetch_current_weather(hours_to_fetch=self.past_hours_to_fetch)
        recent_level = self.level_provider.fetch_recent_level(hours_to_fetch=self.num_recent_samples)

        self._all_current = (current_weather, recent_level)

    def update_for_inference(self):
        self._get_all_current()

    @property
    def all_historical(self):
        if self._all_historical == (None, None):
            self._get_all_historical()

        return self._all_historical

    def _get_all_historical(self):
        historical_weather = self.weather_provider.fetch_historical_weather()
        historical_level = self.level_provider.fetch_historical_level()
        self._all_historical = (historical_weather, historical_level)
