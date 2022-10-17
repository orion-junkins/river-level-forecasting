class Coordinate:
    """
    Basic 2-space coordinate
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return ("(" + str(self.x) + ", " + str(self.y) + ")")

        
    def __str__(self):
        return ("(" + str(self.x) + ", " + str(self.y) + ")")
