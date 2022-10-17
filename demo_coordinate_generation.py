# This file is an interactive, experimental space for generating data site coordinates 
#%%
import os
from forecasting.data_fetching_utilities.coordinate import Coordinate
from forecasting.data_fetching_utilities.location_generator import LocationGenerator

#%%
# Thse coordinates generally represent a bounding box around the Illinois catchment above the gauge near Kerby, OR.
bottom_left = Coordinate(-123.822, 41.883)
upper_right = Coordinate(-123.349, 42.236)

# Desired distance between points
# currently in degrees. Some approximation/conversion for miles would be nice.
separation_distance = 0.05 

#%%
# Generate locations
imposer = LocationGenerator(bottom_left, upper_right, separation_distance)
print(imposer.coordinates)

#%%
# Build a KML file if desired
out_dir = "KML_demos"
os.makedirs(out_dir, exist_ok=True)
filepath = os.path.join(out_dir, "illinois-kerby.KML")

imposer.save_to_kml(filepath=filepath)
# %%
