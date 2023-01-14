import math
import os

import simplekml

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate


class LocationGenerator:
    """
    Driver class for generating locations given the bottom left and top right coordinates of a rectangular area.
    """
    def __init__(self, bottom_left, top_right, separation_degrees=0.25, precision=0.25):
        """Create a Location Generator given the bottom left and top right corners of a rectangular area. Bottom left point will be floored to the nearest rounded value according to provided precision. Top right point will be ceilinged to the nearest rounded value according to provided precision.

        Please Note, the logic of this class will not function if the area enclosed between bottom_left and top_right spans across the meridian.

        Args:
            bottom_left (Coordinate): Bottom left bounding coordinate.
            top_right (Coordinate): Top Right bounding coordinate.
            separation_degrees (float, optional): How far apart points should be in degrees. Defaults to 0.25.
            precision (float, optional): How precise coordinate values should be. Defaults to 0.25.

        Raises:
            ValueError: If bottom_left and top_right do not define a valid region.
        """
        self.separation_degrees = separation_degrees
        self.precision = precision
        self.lon_min = math.floor(bottom_left.lon/self.precision) * self.precision
        self.lat_min = math.floor(bottom_left.lat/self.precision) * self.precision
        self.lon_max = math.ceil(top_right.lon/self.precision) * self.precision
        self.lat_max = math.ceil(top_right.lat/self.precision) * self.precision

        if not (self.separation_degrees > 0.0):
            raise ValueError("separation_degrees must be postive")
        if not (self.lat_min < self.lat_max):
            raise ValueError("lat_min must be less than lat_max")
        if not (self.lon_min < self.lon_max):
            raise ValueError("lon_min must be less than lon_max")

        self.coordinates = self._get_coordinates()

    @property
    def bottom_left(self):
        """The coordinate of the bottom left corner of the bounding box.

        Returns:
            Coordinate: The bottom left coordinate.
        """
        return Coordinate(lon=self.lon_min, lat=self.lat_min)

    @property
    def top_left(self):
        """The coordinate of the top left corner of the bounding box.

        Returns:
            Coordinate: The top left coordinate.
        """
        return Coordinate(lon=self.lon_max, lat=self.lat_min)

    @property
    def bottom_right(self):
        """The coordinate of the bottom right corner of the bounding box.

        Returns:
            Coordinate: The bottom right coordinate.
        """
        return Coordinate(lon=self.lon_min, lat=self.lat_max)

    @property
    def top_right(self):
        """The coordinate of the top right corner of the bounding box.

        Returns:
            Coordinate: The top right coordinate.
        """
        return Coordinate(lon=self.lon_max, lat=self.lat_max)

    def _coordinate_lons(self):
        """The longitude values for the generated coordinates.

        Returns:
            list[float]: The longitude values for the generated coordinates.
        """
        lons = []
        lon_start = self.lon_min
        lon_end = self.lon_max
        while lon_start < lon_end + (self.precision/2):
            rounded_lon = round((lon_start/(self.precision))*self.precision, 4)
            lons.append(rounded_lon)
            lon_start += self.separation_degrees

        return lons

    def _coordinate_lats(self):
        """The latitude values for the generated coordinates.

        Returns:
            list[float]: The latitude values for the generated coordinates.
        """
        lats = []
        lat_start = self.lat_min
        lat_end = self.lat_max
        while lat_start < lat_end + (self.precision/2):
            rounded_lat = round((lat_start/(self.precision))*self.precision, 4)
            lats.append(rounded_lat)
            lat_start += self.separation_degrees

        return lats

    def _get_coordinates(self):
        """Generates coordinates and returns them.

        Returns:
            list[Coordinate]: Generated coordinates.
        """
        coordinates = []
        for lat in self._coordinate_lats():
            for lon in self._coordinate_lons():
                coordinate = Coordinate(lon=lon, lat=lat)
                coordinates.append(coordinate)
        return coordinates

    def save_to_kml(self, filepath=os.path.join('TEST_KML.kml'), render_bounding_box=True):
        """Save the generated coordinates to a kml file. Each coordinate will have an integer id and a name corresponding to the string repr of the Coordinate instance.

        Args:
            filepath (os.path, optional): Destination path to which KML will be saved. Must end in ".kml". Defaults to os.path.join('TEST_KML.kml').
            render_bounding_box (bool, optional): Controls whether a bounding box is rendered to show max and min lat and lon. Defaults to True.
        """
        kml = simplekml.Kml()
        for coordinate in self.coordinates:
            kml.newpoint(name=str(coordinate), coords=[tuple(coordinate)])

        if render_bounding_box:
            linestring = kml.newlinestring(name="bounding_box")
            linestring.coords = [tuple(self.bottom_left),
                                 tuple(self.bottom_right),
                                 tuple(self.top_right),
                                 tuple(self.top_left),
                                 tuple(self.bottom_left)]

        kml.save(filepath)
