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

When adding a new dependency also add a short comment in the yml stating where or for what purpose it is being used. This will make any future dependency issues easier to resolve.

# Dependency Update
Following changes to the environment.yml file, run the following command to update your local environment to match the dependency list specified in `environment.yml`.
```
conda env update -f environment.yml
```

# Install Package Locally

From the root directory of the repo run the following command:
```
pip install -e .
```

This will install the package as 'editable' meaning that any changes to the repo will automatically be reflected the next time you import a module. You will not have to reinstall the package.

# Testing
This repository uses Pytest for unit testing. In order to run all tests, from the root directory, run the following command:
```
pytest
```

To run a specific test file (or subset of test files), run the following command:
```
pytest tests\forecasting\path\to\test\file\or\directory
```

Some tests involve interactions with external APIs, or internal AWS S3 buckets. These tests act as higher level tests to ensure that integration with external services is functioning, but they are slow, and thus are not run by default. In order to run these tests, add the `--runslow` flag as follows:
```
pytest --runslow 
```
or 

```
pytest --runslow tests\forecasting\path\to\test\file\or\directory
```