# River Level Forecasting
Exploring various neural network architectures for river level forecasting
Research conducted by Orion Junkins under Dr. Patrick Donnelly

This repository seeks to compare various methodologies for forecasting multivariate time series data. To demonstrate the efficacy of various methods, each model type can be used to forecast river flow based on a set of weather data sources spread throughout the river's tributaries. 

Using a model trained on historical data, and a snapshot of recent/forecasted weather, forecasts can be generated for a given river gauge.

The accuracy of these forecasts can be compared internally (ie. between model types), and externally (ie against USGS statistically modeled forecasts)

# Dependency Install
All required dependencies are managed through an Anaconda environment. Setup Anaconda and familiarize yourself with environment management if needed.
The needed environment is defined in `environment.yml`
1) Create a new Conda Environment from this yml definition. This will create a new conda environment named 'river-level'
```
conda env create -f environment.yml
```

2) Activate the environment
```
conda activate river-level
```

# Modifying Dependency List
When experimenting, feel free to change package versions and add new packages as needed. If you push a change that relies on a dependency change/addition, be sure to update `environment.yml` with the correct package(s) accordingly.

# Startup
The reset of the README will serve as a high level walkthrough of the core forecasting code. 

Most relevant functionality lives in `./forecasting/`

## 1) Data Fetching
Weather and level data must both be fetched.
### a) Weather Data
Weather data comes from the OpenWeatherAPI.
Explore [./forecasting/data_fetching_utilities/weather.py](./forecasting/data_fetching_utilities/weather.py) to see how this data is fetched and cleaned.

### b) Level Data
Level data comes from [dataretrieval.nwis](https://github.com/USGS-python/dataretrieval), a lovely python package that allows direct querying of nwis data. Explore [./forecasting/data_fetching_utilities/weather.py](./forecasting/data_fetching_utilities/weather.py) to see how this is used.

## 2) Building 'CatchmentData' Instances
Inspect [./rebuild_catchment.py](rebuild_catchment.py). This builds a CatchmentData instance. Inspect [forecasting/catchment_data.py](forecasting/catchment_data.py). 

The resulting CatchmentData instance is pickled and stored. From here on, only the pickle file is needed for training - everything needed lives there.

## 3) Training
Inspect [./train.py](./train.py). This file walks through the procedure of training a model from a CatchmentData instance. This script primarily leverages the Forecaster class from [./forecasting/forecaster.py](./forecasting/forecaster.py). Go through this class to understand what it can and can't do.

## 4) Inference
The forecaster is now able to make predictions (the final print statement from train.py should demonstrate this). Forecasts can be generated and viewed locally, but, for production, we will want to push them to AWS. Explore [aws_dispatcher.py](./aws_dispatcher.py), [rebuild_current_forecast.py](./rebuild_current_forecast.py) and [rebuild_historical_forecast.py](./rebuild_historical_forecast.py) to understand this process.

## 5) Miscellaneous
Some other stuff also lives in this repo. Some useful, some less so. Some should be removed or re-homed.
### a) server_scripts
These are aditional scripts for automation on Patrick's server. Ignore them for now, they are a bit messy but will be useful starting points if/when we start leveraging the DGX2.

### b) data
Data lives here. Ideally the large files should be stored somewhere else, but this dir works for now. Weather locations for each river are also listed here.

### c) darts_custom
This was a failed attempt to extend Darts to allow for multiple different X Covariate sets to be used in an ensemble. It seemed to work, but gave awful results and I never quite got it working. Gave up and went with a naive version instead but this code may be worth coming back to...