# River Level Forecasting
Exploring various neural network architectures for river level forecasting
Research conducted by Orion Junkins under Dr. Patrick Donnelly

This repository seeks to compare various methodologies for forecasting multivariate time series data. To demonstrate the efficacy of various methods, each model type can be used to forecast river flow based on a set of weather data sources spread throughout the river's tributaries. 

Using a model trained on historical data, and a snapshot of recent/forecasted weather, forecasts can be generated for a given river gauge.

The accuracy of these forecasts can be compared internally (ie. between model types), and externally (ie against USGS statistically modeled forecasts)

# Concept Proof Quickstart
```
pip install requirements.txt
```
Explore and run cells in concept-proof.ipynb

