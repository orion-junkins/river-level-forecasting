from data.weather_locations import all_weather_locs
class Catchment:
    def __init__(self, name, usgs_gauge_id, level_forecast_url, weather_locs=None) -> None:
        self.name = name
        self.usgs_gauge_id = str(usgs_gauge_id)
        self.level_forecast_url = level_forecast_url
        if weather_locs == None:
            self.weather_locs = all_weather_locs[name]
        else:
            self.weather_locs = weather_locs


illinois_kerby = Catchment(name = "illinois-kerby",
                        usgs_gauge_id="14377100", 
                        level_forecast_url = "https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage=krbo3&output=tabular")