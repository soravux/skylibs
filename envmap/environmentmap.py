import numpy as np
from scipy.ndimage.interpolation import map_coordinates, zoom

from hdrio import imread

from .tetrahedronSolidAngle import tetrahedronSolidAngle
from .projections import *


SUPPORTED_FORMATS = [
    'angular',
    'skyangular',
    'latlong',
    'sphere',
    'cube',
]

ROTATION_FORMATS = [
    'DCM',
    'EA###',    # TODO
    'EV',       # TODO
    'Q',        # TODO
]


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

    def solidAngles(self):
        """Computes the solid angle subtended by each pixel."""

        # Compute coordinates of pixel borders
        cols = np.linspace(0, 1, self.data.shape[1] + 1)
        rows = np.linspace(0, 1, self.data.shape[0] + 1)

        u, v = np.meshgrid(cols, rows)
        dx, dy, dz, _ = self.image2world(u, v)

        # Split each pixel into two triangles and compute the solid angle
        # subtended by the two tetrahedron
        a = np.vstack((dx[:-1,:-1].ravel(), dy[:-1,:-1].ravel(), dz[:-1,:-1].ravel()))
        b = np.vstack((dx[:-1,1:].ravel(), dy[:-1,1:].ravel(), dz[:-1,1:].ravel()))
        c = np.vstack((dx[1:,:-1].ravel(), dy[1:,:-1].ravel(), dz[1:,:-1].ravel()))
        d = np.vstack((dx[1:,1:].ravel(), dy[1:,1:].ravel(), dz[1:,1:].ravel()))
        omega = tetrahedronSolidAngle(a, b, c)
        omega += tetrahedronSolidAngle(a, b, d)
        
        # Get pixel center coordinates

        _, _, _, valid = self.worldCoordinates()
        omega[~valid.ravel()] = np.nan
        
        return omega.reshape(self.data.shape[0:2])

    def imageCoordinates(self):
        """Returns the (u, v) coordinates for each pixel center."""
        cols = np.linspace(0, 1, self.data.shape[1]*2 + 1)
        rows = np.linspace(0, 1, self.data.shape[0]*2 + 1)

        cols = cols[1::2]
        rows = rows[1::2]

        return [d.astype('float32') for d in np.meshgrid(cols, rows)]

    def worldCoordinates(self):
        """Returns the (x, y, z) world coordinates for each pixel center."""
        u, v = self.imageCoordinates()
        x, y, z, valid = self.image2world(u, v)
        return x, y, z, valid

    def image2world(self, u, v):
        """Returns the (x, y, z) coordinates in the [-1, 1] interval."""
        func = {
            'angular': angular2world,
            'skyangular': skyangular2world,
            'latlong': latlong2world,
            'sphere': sphere2world,
            'cube': cube2world,
        }.get(self.format_)
        return func(u, v)

    def world2image(self, x, y, z):
        """Returns the (u, v) coordinates (in the [0, 1] interval)."""
        func = {
            'angular': world2angular,
            'skyangular': world2skyangular,
            'latlong': world2latlong,
            'sphere': world2sphere,
            'cube': world2cube,
        }.get(self.format_)
        return func(x, y, z)

    def interpolate(self, u, v, valid, method='linear'):
        """"Interpolate to get the desired pixel values."""
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
            interpdata = map_coordinates(source[:,:,c], target, cval=np.nan, order=1)
            data[:,:,c] = interpdata.reshape(data.shape[0], data.shape[1])
        self.data = data

        # In original: valid &= ~isnan(data)...
        # I haven't included it here because it may mask potential problems...
        self.setBackgroundColor(self.backgroundColor, valid)

        return self

    def setBackgroundColor(self, color, valid):
        """Sets the area defined by valid to color."""
        assert valid.dtype == 'bool', "`valid` must be a boolean array."
        assert valid.shape[:2] == self.data.shape[:2], "`valid` must be the same size as the EnvironmentMap."

        self.backgroundColor = np.asarray(color)

        for c in range(self.data.shape[2]):
            self.data[:,:,c][np.invert(valid)] = self.backgroundColor[c]

        return self

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
        self.format_ = targetFormat.lower()
        self.interpolate(u, v, valid)

        return self

    def rotate(self, format, input_):
        """
        Rotate the environment map.

        :param format: Rotation type
        :param input: Rotation information (currently only 3x3 numpy matrix)
        """
        assert format.upper() in ROTATION_FORMATS, "Unknown rotation type '{}'".format(format)
        dx, dy, dz, valid = self.worldCoordinates()

        ptR = np.dot(input_, np.vstack((dx.flatten(), dy.flatten(), dz.flatten())))
        dx, dy, dz = ptR[0].reshape(dx.shape), ptR[1].reshape(dy.shape), ptR[2].reshape(dz.shape)

        dx[dx < -1.] = -1.
        dy[dy < -1.] = -1.
        dz[dz < -1.] = -1.
        dx[dx > 1.] = 1.
        dy[dy > 1.] = 1.
        dz[dz > 1.] = 1.

        u, v = self.world2image(dx, dy, dz)
        self.interpolate(u, v, valid)

        return self

    def resize(self, targetSize, order=1):
        """
        Resize the current environnement map to targetSize.
        targetSize can be a tuple or a single number, in which case the same factor is assumed
        for both u and v.
        If 0 < targetSize < 1, treat it as a ratio
        If targetSize > 1, treat it as new dimensions to use
        """
        if not isinstance(targetSize, tuple):
            targetSize = (targetSize, targetSize)

        _size = []
        for i in range(2):
            _size.append(targetSize[i] / self.data.shape[i] if targetSize[i] > 1. else targetSize[i])

        if len(self.data.shape) > 2:
            _size.append(1.0)   # To ensure we do not "scale" the color axis...

        self.data = zoom(self.data, _size, order=order)
        return self

    def toIntensity(self):
        """
        Returns intensity-version of the environment map.
        This function assumes the CCIR 601 standard to perform internsity conversion.
        """
        assert len(self.data.shape) == 3, "Data should be 3 dimensions"

        if self.data.shape[2] != 3:
            print("Envmap doesn't have 3 channels. This function won't do anything.")
        else:
            self.data = 0.299 * self.data[...,0] + 0.587 * self.data[...,1] + 0.114 * self.data[...,2]
        return self
