import os
import simplekml

from .coordinate import Coordinate

class LocationGenerator:
    """
    Driver class for generating locations given the bottom left and top right coordinates of a rectangular area.
    """
    def __init__(self, bottom_left, top_right, separation_distance=0.025):
        self.lon_min = bottom_left.lon
        self.lat_min = bottom_left.lat
        self.lon_max = top_right.lon
        self.lat_max = top_right.lat
        self.separation_distance = separation_distance

        assert(self.separation_distance > 0.0)
        assert(self.lat_min < self.lat_max)
        assert(self.lon_min < self.lon_max)

        self.coordinates = self._get_coordinates()

        
    @property
    def lon_excess(self):
        return (0.5 * (self.lon_max - self.lon_min) % self.separation_distance)
    
    
    @property
    def lat_excess(self):
        return (0.5 * (self.lat_max - self.lat_min) % self.separation_distance)
    

    def _coordinate_lons(self):
        lons = []
        lon_start = self.lon_min + self.lon_excess
        lon_end = self.lon_max - self.lon_excess

        while lon_start < lon_end:
            lons.append(lon_start)
            lon_start += self.separation_distance

        lons.append(lon_end)

        return lons


    def _coordinate_lats(self):
        lats = []
        lat_start = self.lat_min + self.lat_excess
        lat_end = self.lat_max - self.lat_excess

        while lat_start < lat_end:
            lats.append(lat_start)
            lat_start += self.separation_distance

        lats.append(lat_end)
        return lats


    def _get_coordinates(self):
        coordinates = []
        for lat in self._coordinate_lats():
            for lon in self._coordinate_lons():
                coordinate = Coordinate(lon, lat)
                coordinates.append(coordinate)
        return coordinates


    def save_to_kml(self, filepath=os.path.join('TEST_KML.kml'), render_bounding_box=True):
        kml=simplekml.Kml()
        for coordinate in self.coordinates:
            kml.newpoint(name=str(coordinate), coords=[(coordinate.lon,coordinate.lat)])

        if render_bounding_box:
            linestring = kml.newlinestring(name="bound")
            linestring.coords = [(self.lon_min, self.lat_min),(self.lon_min, self.lat_max), (self.lon_max, self.lat_max), (self.lon_max, self.lat_min), (self.lon_min, self.lat_min)]

        kml.save(filepath)