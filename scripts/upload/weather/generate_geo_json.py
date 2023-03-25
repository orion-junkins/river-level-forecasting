# Helper script for selecting locations. Assumes that the specified "KML_DIR_PATH" exists containing catchment definition KML files as created with the "Drainage Area Delineation" provided with the WATERS Geo Dataset. These KML files should be named "[Gauge_name] - #[Gauge_id]". For example: "CANYON CREEK NEAR AMBOY, WA - #14219000.kml" This script will produce a GeoJSON file at the specified output path containing a list of the coordinates that define the drainage, the size of the drainage, the bounding coordinates for the grid imposed over the drainage, and a list of coordinates that make up the imposed grid itself.
# If only coordinate values are needed (most use cases) set COORDINATES_ONLY to True to exclude storage intensive catchment definitions.
import json
import os
import re
import sys

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate
from rlf.forecasting.data_fetching_utilities.location_generator import \
    LocationGenerator

# Parse command line args
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

# Define the input and output paths
if len(args) == 2:
    KML_DIR_PATH = args[0]
    OUTPUT_FILEPATH = args[1]
else:
    raise ValueError(f'Usage: {sys.argv[0]} (-long) KML_DIR_PATH OUTPUT_FILEPATH')

# Specify if only coordinates are needed (creates a much smaller file when True)
COORDINATES_ONLY = not ("-long" in opts)


def process_kml(path, gauge_id, gauge_name):
    """Process a single kml file at the given path. Produce three dictionaries that define the selected points, the bounding box of those points, and the drainage delineation in geoJSON format.

    Args:
        path (str): Path to the KML file.
        gauge_id (str): The USGS id for the gauge.
        gauge_name (str): The USGS name for the gauge.

    Returns:
        tuple(dict, dict, dict): Tuple of three dictionaries (selected_points_feature, bbox_feature, catchment_delineation_feature).
    """
    # Read the file
    with open(path) as f:
        doc = f.read()

    # Isolate the catchment area size, or default to -1 if no match is found
    area_result = re.search(
        r'"total_areasqkm">\s*<value>\s*([0-9]*\.[0-9]*)\s*<\/value>', doc)
    if area_result:
        area_km = float(area_result.group(1))
    else:
        area_km = -1

    # Isolate the coordinates that define the outer boundary (ignores inner boundaries)
    outer_boundary = re.search(
        r'<outerBoundaryIs>([\W\w]*)</outerBoundaryIs>', doc).group(1)
    coord_results = re.findall(
        r'(-{0,1}[0-9]{1,3}\.[0-9]+),(-{0,1}[0-9]{1,3}\.[0-9]+),0', outer_boundary)

    # Create lists of lats, lons and drainage delineation coordinates (in the form [lon, lat])
    lats = []
    lons = []
    drainage_delineation_coordinates = []
    for result in coord_results:
        lon, lat = result
        lons.append(float(lon))
        lats.append(float(lat))
        coord = [float(lon), float(lat)]
        drainage_delineation_coordinates.append(coord)

    # Identify bounding coordinates
    bottom_left_bounding_coordinate = Coordinate(min(lons), lat=min(lats))
    top_right_bounding_coordinate = Coordinate(lon=max(lons), lat=max(lats))

    # Generate a grid of coordinates within the identified bounds
    location_generator = LocationGenerator(
        bottom_left_bounding_coordinate, top_right_bounding_coordinate)

    imposed_grid_coordinates_lists = location_generator.coordinates

    # Convert the list of Coordinates into a simple list of lists in the form [lon, lat]
    imposed_grid_coordinates = []
    for coord_list in imposed_grid_coordinates_lists:
        coord = [float(coord_list[0]), float(coord_list[1])]
        imposed_grid_coordinates.append(coord)

    # Define a geoJSON feature for the selected points
    selected_points_feature = {
        "type": "Feature",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "properties": {"gauge_id": gauge_id, "gauge_name": gauge_name, "area_sq_km": area_km},
        "geometry": {
            "type": "MultiPoint",
            "coordinates": imposed_grid_coordinates
        }
    }

    # Define a geoJSON feature for the bounding box
    bbox_feature = {
        "type": "Feature",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "properties": {"gauge_id": gauge_id, "gauge_name": gauge_name, "area_sq_km": area_km},
        "geometry": {
            "type": "LineString",
            "coordinates": [[min(lons), min(lats)], [min(lons), max(lats)], [max(lons), max(lats)], [max(lons), min(lats)], [min(lons), min(lats)]]
        }
    }

    # Define a geoJSON feature for the drainage delineation
    catchment_delineation_feature = {
        "type": "Feature",
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "properties": {"gauge_id": gauge_id, "gauge_name": gauge_name, "area_sq_km": area_km},
        "geometry": {
            "type": "Polygon",
            "coordinates": [drainage_delineation_coordinates]
        }
    }
    return (selected_points_feature, bbox_feature, catchment_delineation_feature)


# Create a basic dictionary for geoJSON features
output_data = {"type": "FeatureCollection", "features": []}

# For every kml file in the provided directory,
for filename in os.listdir(KML_DIR_PATH):
    filepath = os.path.join(KML_DIR_PATH, filename)
    if os.path.isfile(filepath) and filepath.endswith(".kml"):
        print("Processing: " + filename)

        # Parse the gauge name and gauge id from the filename
        gauge_description = filename[:-4]
        result = re.search(r'(.*) - #([0-9]{5,15})', gauge_description)

        if result is None:
            print("\tUnable to parse filename for gauge details. Skipping.")
            continue
        
        gauge_name = result.group(1)
        gauge_id = result.group(2)

        # Process the KML into geoJSON dictionaries
        selected_points_feature, bbox_feature, catchment_delineation_feature = process_kml(
            filepath,
            gauge_id=gauge_id, gauge_name=gauge_name)

        # Append these geoJSON dicts as features
        output_data["features"].append(selected_points_feature)
        if not COORDINATES_ONLY:
            output_data["features"].append(bbox_feature)
            output_data["features"].append(catchment_delineation_feature)

# Output the JSON file to the specified path
jsonString = json.dumps(output_data, indent=4)
with open(OUTPUT_FILEPATH, "w") as outfile:
    outfile.write(jsonString)
