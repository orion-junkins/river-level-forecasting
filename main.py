#%%
from src.model_builders import *
from src.forecast_site import ForecastSite
from src.forecaster import *
fsite = ForecastSite("14377100", ["historical_weather_data/illinois-kerby/41.980609,-123.613583.csv"])
# %%


# Example usage: 
# Based on today's date, fetch all relevant water data
mck_vida_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=14162500&referred_module=sw&period=&begin_date=1988-10-12&end_date=' + yesterday()

# Fetch weather & SWE data for Mckenzie basin. Atuomatically retrieves all data up to present.
mck1_weather_url ='https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/619:OR:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value'

mck2_weather_url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/719:OR:SNTL%7Cid=%22%22%7Cname/1980-11-03,2022-01-18/WTEQ::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value?fitToScreen=false'

# %%
frcstr = Forecaster([mck1_weather_url, mck2_weather_url], mck_vida_url, build_conv_model)

frcstr_backup = frcstr
#%%
frcstr.fit()

#%%
frcstr.forecast_all()
print(frcstr.forecasted_levels)