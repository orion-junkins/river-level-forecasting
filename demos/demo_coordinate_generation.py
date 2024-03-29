# This file is an interactive, experimental space for generating data site coordinates
# %%
import os

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator

# %%
# Thse coordinates generally represent a bounding box around the Illinois catchment above the gauge near Kerby, OR.
bottom_left = Coordinate(lon=-123.822, lat=41.883)
upper_right = Coordinate(lon=-123.349, lat=42.236)

# %%
# Generate locations
location_generator = LocationGenerator(bottom_left, upper_right)
print(location_generator.coordinates)

# %%
# Build a KML file if desired
out_dir = "KML_demos"
os.makedirs(out_dir, exist_ok=True)
filepath = os.path.join(out_dir, "illinois-kerby.KML")

location_generator.save_to_kml(filepath=filepath)
