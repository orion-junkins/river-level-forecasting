#%%
from datetime import datetime, timedelta
from dataset import *
#%%
def yesterday() -> str:
    """
    Helper function for data retrieval
    Returns:
        yesterday (string): yesterdays date in the format "%Y-%m-%d"
    """
    yesterday = datetime.today()  - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")
    return yesterday

#%% 
# Example usage: 
# Based on today's date, fetch all relevant water data
mck_vida_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=14162500&referred_module=sw&period=&begin_date=1988-10-12&end_date=' + yesterday()

# Fetch weather & SWE data for Mckenzie basin. Atuomatically retrieves all data up to present.
mck1_weather_url ='https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/619:OR:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value,PREC::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value'

mck2_weather_url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/719:OR:SNTL%7Cid=%22%22%7Cname/1980-11-03,2022-01-08/WTEQ::value,TMAX::value,TMIN::value,TAVG::value,PRCP::value?fitToScreen=false'

ds = Dataset([mck1_weather_url, mck2_weather_url], mck_vida_url)
# %%
