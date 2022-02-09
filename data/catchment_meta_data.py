from data.weather_locations import all_weather_locs

class Catchment:
    def __init__(self, name, usgs_gauge_id, level_forecast_url=None, weather_locs=None) -> None:
        self.name = name
        self.usgs_gauge_id = str(usgs_gauge_id)
        self.level_forecast_url = level_forecast_url
        if weather_locs == None:
            self.weather_locs = all_weather_locs[name]
        else:
            self.weather_locs = weather_locs


icicle_abv_snow = Catchment(name = "icicle-abv-snow",
                        usgs_gauge_id = "12458000")

white_salmon_underwood = Catchment(name = "white-salmon-underwood",
                        usgs_gauge_id = "14123500")

cispus_abv_yellowjacket = Catchment(name = "cispus-abv-yellowjacket",
                        usgs_gauge_id = "14231900",
                        level_forecast_url = "https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage=ciyw1&output=tabular")

illinois_kerby = Catchment(name = "illinois-kerby",
                        usgs_gauge_id="14377100", 
                        level_forecast_url = "https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage=krbo3&output=tabular")


mckenzie_vida = Catchment(name = "mckenzie-vida",
                        usgs_gauge_id="14162500",
                        level_forecast_url = "https://water.weather.gov/ahps2/hydrograph_to_xml.php?gage=vido3&output=tabular")