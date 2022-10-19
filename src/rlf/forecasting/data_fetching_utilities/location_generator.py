import os

from geopy import distance
import simplekml

from rlf.forecasting.data_fetching_utilities.coordinate import Coordinate


class LocationGenerator:
    """
    Driver class for generating locations given the bottom left and top right coordinates of a rectangular area.
    """
    def __init__(self, bottom_left, top_right, separation_distance_km=25):
        self.lon_min = bottom_left.lon
        self.lat_min = bottom_left.lat
        self.lon_max = top_right.lon
        self.lat_max = top_right.lat
        self.separation_distance_km = separation_distance_km

        if not (self.separation_distance_km > 0.0):
            raise ValueError("separation_distance_km must be postive")
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
        return Coordinate(self.lat_min, self.lon_min)

    @property
    def top_left(self):
        """The coordinate of the top left corner of the bounding box.

        Returns:
            Coordinate: The top left coordinate.
        """
        return Coordinate(self.lat_max, self.lon_min)

    @property
    def bottom_right(self):
        """The coordinate of the bottom right corner of the bounding box.

        Returns:
            Coordinate: The bottom right coordinate.
        """
        return Coordinate(self.lat_min, self.lon_max)

    @property
    def top_right(self):
        """The coordinate of the top right corner of the bounding box.

        Returns:
            Coordinate: The top right coordinate.
        """
        return Coordinate(self.lat_max, self.lon_max)

    @property
    def lat_separation(self):
        """The latitude separation between locations in degrees. Approximated based on the provided separation distance in kilometers.

        Returns:
            float: The latitude separation between locations in degrees.
        """
        # Find the distance between lat min and lat max in degrees
        lat_dif_deg = self.lat_max - self.lat_min

        # Find the average distance between lat min and lat max in kilometers
        lat_dif_top_km = distance.distance(tuple(self.top_left), tuple(self.top_right)).km
        lat_dif_bottom_km = distance.distance(tuple(self.bottom_left), tuple(self.bottom_right)).km
        average_lat_dif_km = (lat_dif_top_km + lat_dif_bottom_km) / 2

        # Convert the separation distance from kilometers to degrees
        seperation_degrees = (lat_dif_deg * self.separation_distance_km) / average_lat_dif_km

        return seperation_degrees

    @property
    def lon_separation(self):
        """The longitude separation between locations in degrees. Approximated based on the provided separation distance in kilometers.

        Returns:
            float: The longitude separation between locations in degrees.
        """
        # Find the distance between lon min and lon max in degrees
        lon_dif_deg = self.lon_max - self.lon_min

        # Find the average distance between lat min and lat max in kilometers
        lon_dif_left_km = distance.distance(tuple(self.top_left), tuple(self.bottom_left)).km
        lon_dif_right_km = distance.distance(tuple(self.top_right), tuple(self.bottom_right)).km
        average_lon_dif_km = (lon_dif_left_km + lon_dif_right_km) / 2

        # Convert the separation distance from kilometers to degrees
        seperation_degrees = (lon_dif_deg * self.separation_distance_km) / average_lon_dif_km

        return seperation_degrees

    @property
    def lat_excess(self):
        """The amount of excess latitudinal distance in degrees that cannot be evenly divided by lat_separation.

        Returns:
            float: The amount of excess distance in degrees.
        """
        return (0.5 * (self.lat_max - self.lat_min) % self.lat_separation)

    @property
    def lon_excess(self):
        """The amount of excess longitudinal distance in degrees that cannot be evenly divided by lon_separation.

        Returns:
            float: The amount of excess distance in degrees.
        """
        return (0.5 * (self.lon_max - self.lon_min) % self.lon_separation)

    def _coordinate_lons(self):
        """The longitude values for the generated coordinates.

        Returns:
            list[float]: The longitude values for the generated coordinates.
        """
        lons = []
        lon_start = self.lon_min + self.lon_excess
        lon_end = self.lon_max - self.lon_excess

        while lon_start < lon_end:
            lons.append(lon_start)
            lon_start += self.lon_separation

        lons.append(lon_end)

        return lons

    def _coordinate_lats(self):
        """The latitude values for the generated coordinates.

        Returns:
            list[float]: The latitude values for the generated coordinates.
        """
        lats = []
        lat_start = self.lat_min + self.lat_excess
        lat_end = self.lat_max - self.lat_excess

        while lat_start < lat_end:
            lats.append(lat_start)
            lat_start += self.lat_separation

        lats.append(lat_end)
        return lats

    def _get_coordinates(self):
        """Generates coordinates and returns them.

        Returns:
            list[Coordinate]: Generated coordinates.
        """
        coordinates = []
        for lat in self._coordinate_lats():
            for lon in self._coordinate_lons():
                coordinate = Coordinate(lat, lon)
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
            kml.newpoint(name=str(coordinate), coords=[tuple(reversed(tuple(coordinate)))])

        if render_bounding_box:
            linestring = kml.newlinestring(name="bounding_box")
            linestring.coords = [tuple(reversed(tuple(self.bottom_left))),
                                 tuple(reversed(tuple(self.bottom_right))),
                                 tuple(reversed(tuple(self.top_right))),
                                 tuple(reversed(tuple(self.top_left))),
                                 tuple(reversed(tuple(self.bottom_left)))]

        kml.save(filepath)
