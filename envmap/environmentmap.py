import numpy as np

SUPPORTED_FORMATS = [
    'angular',
    'latlong',
]

eps = 2**-52


class EnvironmentMap:
    def __init__(self, im, format_):
        """
        Creates an EnvironmentMap.

        :param im: Image to be converted to an EnvironmentMap
        :param format_: EnvironmentMap format. Can be `Angular`, ...
        :type im: float, numpy array
        :type format_: string
        """
        assert format_.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(format_))

        self.im = im
        self.format_ = format_

    def imageCoordinates(self):
        """Returns the (u, v) coordinates for each pixel center."""
        cols = np.linspace(0, 1, self.im.shape[1]*2 + 1)
        rows = np.linspace(0, 1, self.im.shape[0]*2 + 1)

        cols = cols[1::2]
        rows = rows[1::2]

        return np.meshgrid(cols, rows)


    def worldCoordinates(self):
        """Returns the (x, y, z) world coordinates for each pixel center."""
        u, v = self.imageCoordinates()
        x, y, z, valid = self.image2world(u, v)
        return x, y, z, valid

    def image2world(self, u, v):
        pass

    def world2image(self, x, y, z):
        func = {
            'angular': self.world2angular,
            'latlong': self.world2latlong,
        }.get(self.format_)
        return func(x, y, z)

    def interpolate(self, u, v, valid):
        pass

    def convertTo(self, targetFormat, targetDim):
        """
        Convert to another format.

        :param targetFormat: Target format.
        :param targetDim: Target dimensions.
        :type targetFormat: string
        :type targetFormat: float, array
        """
        eTmp = EnvironmentMap(np.zeros(self.im.shape), targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        u, v = self.world2image(dx, dy, dz)
        self.format_ = targetFormat
        self.interpolate(u, v, valid)

    def world2latlong(self, x, y, z):
        """Get the (u, v) coordinates of the point defined by (x, y, z) for
        a latitude-longitude map."""
        u = 1 + (1/pi) * np.arctan2(x, -z);
        v = (1/pi) * np.arccos(y);
        # because we want [0,1] interval
        u = u/2;
        return u, v

    def world2angular(self, x, y, z):
        """Get the (u, v) coordinates of the point defined by (x, y, z) for
        an angular map."""
        # world -> angular
        denum = (2*np.pi*np.sqrt(x**2 + y**2))+eps;
        rAngular = np.arccos(-z) / denum;
        v = 1/2-rAngular*y;
        u = 1/2+rAngular*x;
        return u, v
