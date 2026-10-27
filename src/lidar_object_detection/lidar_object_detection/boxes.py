"""3D box detection."""


class Box3D:
    """An oriented 3D bounding box detection in the sensor frame.

    x, y, z    -- box center (m)
    dx, dy, dz -- extents along the box axes (m)
    heading    -- rotation about +z (rad)
    """

    __slots__ = ('class_name', 'score', 'x', 'y', 'z',
                 'dx', 'dy', 'dz', 'heading')

    def __init__(self, class_name, score, x, y, z, dx, dy, dz, heading):
        self.class_name = class_name
        self.score = score
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.heading = heading

    def range(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
