import simplekml
from .coordinate import Coordinate

class LocationGenerator:
    """
    Driver class for grid imposition given the bottom left, and top right coordinates of an area.
    """
    def __init__(self, bottom_left, top_right, separation_distance=0.025):
        self.x_min = bottom_left.x
        self.y_min = bottom_left.y
        self.x_max = top_right.x
        self.y_max = top_right.y
        self.separation_distance = separation_distance

        self.coordinates = [] # list of coordinates
        self.get_coordinates()
        
    @property
    def x_excess(self):
        return (0.5 * (self.x_max - self.x_min) % self.separation_distance)
    
    @property
    def y_excess(self):
        return (0.5 * (self.y_max - self.y_min) % self.separation_distance)

    @property
    def coordinate_xs(self):
        xs = []
        x_start = self.x_min + self.x_excess
        x_end = self.x_max - self.x_excess

        while x_start < x_end:
            xs.append(x_start)
            x_start += self.separation_distance

        xs.append(x_end)

        return xs

    @property
    def coordinate_ys(self):
        ys = []
        y_start = self.y_min + self.y_excess
        y_end = self.y_max - self.y_excess

        while y_start < y_end:
            ys.append(y_start)
            y_start += self.separation_distance

        ys.append(y_end)
        return ys

    def get_coordinates(self):
        coordinates = []
        for y in self.coordinate_ys:
            for x in self.coordinate_xs:
                coordinate = Coordinate(x, y)
                coordinates.append(coordinate)
        self.coordinates = coordinates


    def build_kml(self, filename='TEST_KML.kml'):
        kml=simplekml.Kml()
        coordinates = self.coordinates
        for coordinate in coordinates:
            kml.newpoint(name=str(coordinate), coords=[(coordinate.x,coordinate.y)])

        linestring = kml.newlinestring(name="bound")
        linestring.coords = [(self.x_min, self.y_min),(self.x_min, self.y_max), (self.x_max, self.y_max), (self.x_max, self.y_min), (self.x_min, self.y_min)]

        kml.save(filename)