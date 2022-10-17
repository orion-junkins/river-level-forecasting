from collections import namedtuple

# Create a namedtuple type, Point
Coordinate = namedtuple("Coordinate", "lat lon")
issubclass(Coordinate, tuple)
