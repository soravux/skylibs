import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from hdrio import imread


SUPPORTED_FORMATS = [
    'angular',
    'skyangular',
    'latlong',
    'sphere',
    'cube', # TODO: Not done!
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

        self.format_ = format_.lower()
        self.backgroundColor = np.array([0, 0, 0])

        if isinstance(im, str):
            # We received the filename
            self.data = imread(im)
        elif isinstance(im, int):
            # We received a single scalar
            if self.format_ == 'latlong':
                self.data = np.zeros((im, im*2))
            elif self.format_ == 'cube':
                self.data = np.zeros((im, round(3/4*im)))
            else:
                self.data = np.zeros((im, im))
        elif type(im).__module__ == np.__name__:
            # We received a numpy array
            self.data = np.asarray(im, dtype='double')
        else:
            raise Exception('Could not understand input. Please prove a '
                            'filename, a size or an image.')

        # Ensure size is valid
        if self.format_ in ['sphere', 'angular', 'skysphere', 'skyangular']:
            assert self.data.shape[0] == self.data.shape[1], (
                "Sphere/Angular formats must have the same width/height")

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
            'skyangular': self.skyangular2world,
            'latlong': self.latlong2world,
            'sphere': self.sphere2world,
        }.get(self.format_)
        return func(u, v)

    def world2image(self, x, y, z):
        """Returns the (u, v) coordinates (in the [0, 1] interval)."""
        func = {
            'angular': self.world2angular,
            'skyangular': self.world2skyangular,
            'latlong': self.world2latlong,
            'sphere': self.world2sphere,
        }.get(self.format_)
        return func(x, y, z)

    def interpolate(self, u, v, valid, method='linear'):
        """"Interpolate to get the desired pixel values."""
        cols, rows = self.imageCoordinates()
        cols = cols[0, :]
        rows = rows[:, 0]
        target = np.vstack((v.flatten()*self.data.shape[0], u.flatten()*self.data.shape[1]))

        # Repeat the first and last rows/columns for interpolation purposes
        h, w, d = self.data.shape
        source = np.empty((h + 2, w + 2, d))
        source[1:-1, 1:-1] = self.data
        source[0,1:-1] = self.data[0,:]; source[0,0] = self.data[0,0]; source[0,-1] = self.data[0,-1]
        source[-1,1:-1] = self.data[-1,:]; source[-1,0] = self.data[-1,0]; source[-1,-1] = self.data[-1,-1]
        source[1:-1,0] = self.data[:,0]
        source[1:-1,-1] = self.data[:,-1]

        data = np.zeros((u.shape[0], u.shape[1], self.data.shape[2]))
        for c in range(self.data.shape[2]):
            interpdata = map_coordinates(source[:,:,c], target, cval=np.nan)
            data[:,:,c] = interpdata.reshape(data.shape[0], data.shape[1])
        self.data = data

        # In original: valid &= ~isnan(data)...
        # I haven't included it here because it may mask potential problems...
        self.setBackgroundColor(self.backgroundColor, valid)

    def setBackgroundColor(self, color, valid):
        """Sets the area defined by valid to color."""
        assert valid.dtype == 'bool', "`valid` must be a boolean array."
        assert valid.shape[:2] == self.data.shape[:2], "`valid` must be the same size as the EnvironmentMap."

        self.backgroundColor = np.asarray(color)

        for c in range(self.data.shape[2]):
            self.data[:,:,c][np.invert(valid)] = self.backgroundColor[c]

    def convertTo(self, targetFormat, targetDim=None):
        """
        Convert to another format.

        :param targetFormat: Target format.
        :param targetDim: Target dimension.
        :type targetFormat: string
        :type targetFormat: integer

.. todo::

    Support targetDim

        """
        assert targetFormat.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(targetFormat))

        if not targetDim:
            # By default, number of rows
            targetDim = self.data.shape[0]

        eTmp = EnvironmentMap(targetDim, targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        u, v = self.world2image(dx, dy, dz)
        self.format_ = targetFormat
        self.interpolate(u, v, valid)

    def world2latlong(self, x, y, z):
        """Get the (u, v) coordinates of the point defined by (x, y, z) for
        a latitude-longitude map."""
        u = 1 + (1/np.pi) * np.arctan2(x, -z)
        v = (1/np.pi) * np.arccos(y)
        # because we want [0,1] interval
        u = u/2
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
        phiAngular = np.pi*np.sqrt((2*u-1)**2 + (2*v-1)**2)

        x = np.sin(phiAngular)*np.cos(thetaAngular)
        y = np.sin(phiAngular)*np.sin(thetaAngular)
        z = -np.cos(phiAngular)

        r = (u-0.5)**2 + (v-0.5)**2
        valid = r <= .25 # .5**2

        return x, y, z, valid

    def skyangular2world(self, u, v):
        """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
        for a sky angular map."""
        # skyangular -> world
        thetaAngular = np.arctan2(-2*v+1, 2*u-1) # azimuth
        phiAngular = np.pi/2*np.sqrt((2*u-1)**2 + (2*v-1)**2) # zenith

        x = np.sin(phiAngular)*np.cos(thetaAngular)
        z = np.sin(phiAngular)*np.sin(thetaAngular)
        y = np.cos(phiAngular)

        r = (u-0.5)**2 + (v-0.5)**2
        valid = r <= .25 # .5^2

        return x, y, z, valid

    def world2skyangular(self, x, y, z):
        """Get the (u, v) coordinates of the point defined by (x, y, z) for
        a sky angular map."""
        # world -> skyangular
        thetaAngular = np.arctan2(x, z) # azimuth
        phiAngular = np.arctan2(np.sqrt(x**2+z**2), y) # zenith

        r = phiAngular/(np.pi/2);

        u = r*np.sin(thetaAngular)/2+1/2
        v = 1/2-r*np.cos(thetaAngular)/2

        return u, v

    def sphere2world(self, u, v):
        """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
        for the sphere map."""
        u = u*2 - 1
        v = v*2 - 1

        # sphere -> world
        r = sqrt(u**2 + v**2)
        theta = np.arctan2(u, -v)

        phi = np.zeros(theta.shape)
        valid = r <= 1
        phi[valid] = 2*np.arcsin(r[valid])

        x = np.sin(phi)*np.sin(theta)
        y = np.sin(phi)*np.cos(theta)
        z = -np.cos(phi)
        return x, y, z, valid

    def world2sphere(self, x, y, z):
        # world -> sphere
        denum = (2*np.sqrt(x**2 + y**2)) + eps
        r = np.sin(.5*np.arccos(-z)) / denum

        u = .5 + r*x
        v = .5 - r*y

        return u, v
