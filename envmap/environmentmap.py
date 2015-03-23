import numpy as np
from scipy.interpolate import griddata

SUPPORTED_FORMATS = [
    'angular',
    'latlong',
]

eps = 2**-52


class EnvironmentMap:
    """
.. todo::

    * Move world2* and *2world to transforms.py

    """
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

        self.data = im
        self.format_ = format_
        self.backgroundColor = np.array([0, 0, 0])

    def imageCoordinates(self):
        """Returns the (u, v) coordinates for each pixel center."""
        cols = np.linspace(0, 1, self.data.shape[1]*2 + 1)
        rows = np.linspace(0, 1, self.data.shape[0]*2 + 1)

        cols = cols[1::2]
        rows = rows[1::2]

        return np.meshgrid(cols, rows)

    def worldCoordinates(self):
        """Returns the (x, y, z) world coordinates for each pixel center."""
        u, v = self.imageCoordinates()
        x, y, z, valid = self.image2world(u, v)
        return x, y, z, valid

    def image2world(self, u, v):
        """Returns the (x, y, z) coordinates in the [-1, 1] interval."""
        func = {
            'angular': self.angular2world,
            'latlong': self.latlong2world,
        }.get(self.format_)
        return func(u, v)

    def world2image(self, x, y, z):
        """Returns the (u, v) coordinates (in the [0, 1] interval)."""
        func = {
            'angular': self.world2angular,
            'latlong': self.world2latlong,
        }.get(self.format_)
        return func(x, y, z)

    def interpolate(self, u, v, valid, method='linear'):
        """"Interpolate to get the desired pixel values.
.. todo::

    Is griddata the best option for this?

"""
        cols, rows = self.imageCoordinates()
        cols = cols[0, :]
        rows = rows[:, 0]
        coords = np.hstack(cols.T, rows)

        data = np.array((u.size, v.size, self.data.shape[2]))
        for c in range(self.data.shape[2]):
            data = griddata(coords, self.data[:,:,c], (u, v), method=method)
        self.data = data

        # In original: valid & ~isnan(data)...
        # I haven't included it here because it may mask potential problems...
        self.setBackgroundColor(self.backgroundColor, valid)

    def setBackgroundColor(self, color, mask):
        """Sets the area defined by mask to color."""
        assert mask.dtype == 'bool', "`mask` must be a boolean array."
        assert mask.shape == self.data.shape, "`mask` must be the same size as the EnvironmentMap."

        self.backgroundColor = np.asarray(color)

        self.data[mask] = self.backgroundColor

    def convertTo(self, targetFormat, targetDim=None):
        """
        Convert to another format.

        :param targetFormat: Target format.
        :param targetDim: Target dimensions.
        :type targetFormat: string
        :type targetFormat: float, array

.. todo::

    Support targetDim

        """
        assert targetFormat.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(targetFormat))

        eTmp = EnvironmentMap(np.zeros(self.data.shape), targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        u, v = self.world2image(dx, dy, dz)
        self.format_ = targetFormat
        self.interpolate(u, v, valid)

    def world2latlong(self, x, y, z):
        """Get the (u, v) coordinates of the point defined by (x, y, z) for
        a latitude-longitude map."""
        u = 1 + (1/pi) * np.arctan2(x, -z)
        v = (1/pi) * np.arccos(y)
        # because we want [0,1] interval
        u = u/2;
        return u, v

    def world2angular(self, x, y, z):
        """Get the (u, v) coordinates of the point defined by (x, y, z) for
        an angular map."""
        # world -> angular
        denum = (2*np.pi*np.sqrt(x**2 + y**2))+eps
        rAngular = np.arccos(-z) / denum
        v = 1/2-rAngular*y
        u = 1/2+rAngular*x
        return u, v

    def latlong2world(self, u, v):
        """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
        for a latlong map."""
        u = u*2

        # lat-long -> world
        thetaLatLong = np.pi*(u-1)
        phiLatLong = np.pi*v

        x = np.sin(phiLatLong)*np.sin(thetaLatLong)
        y = np.cos(phiLatLong)
        z = -np.sin(phiLatLong)*np.cos(thetaLatLong)

        valid = np.ones(x.shape, dtype='bool')
        return x, y, z, valid

    def angular2world(self, u, v):
        """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
        for an angular map."""
        # angular -> world
        thetaAngular = np.arctan2(-2*v+1, 2*u-1)
        phiAngular = np.pi*sqrt((2*u-1)**2 + (2*v-1)**2)

        x = np.sin(phiAngular)*np.cos(thetaAngular)
        y = np.sin(phiAngular)*np.sin(thetaAngular)
        z = -np.cos(phiAngular)

        r = (u-0.5)**2 + (v-0.5)**2
        valid = r <= .25 # .5**2

        return x, y, z, valid
