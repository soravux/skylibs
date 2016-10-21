import hashlib
import os
from copy import deepcopy

import numpy as np
from scipy.ndimage.interpolation import map_coordinates, zoom

from hdrio import imread

from .tetrahedronSolidAngle import tetrahedronSolidAngle
from .projections import *
from .xmlhelper import EnvmapXMLParser


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


#From Dan:
#  I've generated these using the monochromatic albedo values from here:
#  http://agsys.cra-cin.it/tools/solarradiation/help/Albedo.html
#  (they cite some books as references). Since these monochromatic,
#  I got unscaled r, g, b values from internet textures and scaled them
#  so that their mean matches the expected monochromatic albedo. Hopefully
#  this is a valid thing to do.
GROUND_ALBEDOS = {
    "GreenGrass": np.array([ 0.291801, 0.344855, 0.113344 ]).T,
    "FreshSnow": np.array([ 0.797356, 0.835876, 0.916767 ]).T,
    "Asphalt": np.array([ 0.148077, 0.150000, 0.151923 ]).T,
}


class EnvironmentMap:
    def __init__(self, im, format_=None, copy=True, color=True):
        """
        Creates an EnvironmentMap.

        :param im: Image path or data to be converted to an EnvironmentMap, or 
                   the height of an empty EnvironmentMap.
        :param format_: EnvironmentMap format. Can be `Angular`, ...
        :param copy: When a numpy array is given, should it be copied.
        :param color: When providing an integer, create an empty color or
                      grayscale EnvironmentMap.
        :type im: float, numpy array
        :type format_: string
        :type copy: bool
        """
        if not format_ and isinstance(im, str):
            filename = os.path.splitext(im)[0]
            metadata = EnvmapXMLParser("{}.meta.xml".format(filename))
            format_ = metadata.getFormat()

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
                if color:
                    self.data = np.zeros((im, im, 3))
                else:
                    self.data = np.zeros((im, im))
        elif type(im).__module__ == np.__name__:
            # We received a numpy array
            self.data = np.asarray(im, dtype='double')

            if copy:
                self.data = self.data.copy()
        else:
            raise Exception('Could not understand input. Please prove a '
                            'filename, a size or an image.')

        # Ensure size is valid
        if self.format_ in ['sphere', 'angular', 'skysphere', 'skyangular']:
            assert self.data.shape[0] == self.data.shape[1], (
                "Sphere/Angular formats must have the same width/height")
        elif self.format_ == 'latlong':
            assert 2*self.data.shape[0] == self.data.shape[1], (
                "LatLong format width should be twice the height")

    def __hash__(self):
        """Provide a hash on the environment map"""
        h = hashlib.sha1(self.data.view(np.uint8))
        h.update(self.format_.encode('utf-8'))
        return int(h.hexdigest(), 16)

    def copy(self):
        """Returns a copy of the current environment map."""
        return deepcopy(self)

    def solidAngles(self):
        """Computes the solid angle subtended by each pixel."""
        # If already computed, take it
        if hasattr(self, '_solidAngles') and hash(self) == self._solidAngles_hash:
            return self._solidAngles

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

        self._solidAngles = omega.reshape(self.data.shape[0:2])
        self._solidAngles_hash = hash(self)
        return self._solidAngles

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
        if len(self.data.shape) == 2:
            h, w = self.data.shape
            d = 1
        else:
            h, w, d = self.data.shape
        source = np.empty((h + 2, w + 2, d))
        source[1:-1, 1:-1] = self.data
        source[0,1:-1] = self.data[0,:]; source[0,0] = self.data[0,0]; source[0,-1] = self.data[0,-1]
        source[-1,1:-1] = self.data[-1,:]; source[-1,0] = self.data[-1,0]; source[-1,-1] = self.data[-1,-1]
        source[1:-1,0] = self.data[:,0]
        source[1:-1,-1] = self.data[:,-1]

        # To avoid displacement due to the padding
        u += 1./self.data.shape[1]
        v += 1/self.data.shape[0]

        data = np.zeros((u.shape[0], u.shape[1], d))
        for c in range(d):
            map_coordinates(source[:,:,c], target, output=data[:,:,c].reshape(-1), cval=np.nan, order=1, prefilter=False)
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

        grayscale = len(self.data.shape) == 2

        if grayscale:
            self.data[:,:][np.invert(valid)] = self.backgroundColor.dot(np.array([0.299, 0.587, 0.114]).T)
        else:
            nb_channels = self.data.shape[2]
            for c in range(nb_channels):
                self.data[:,:,c][np.invert(valid)] = self.backgroundColor[c]

        return self

    def convertTo(self, targetFormat, targetDim=None):
        """
        Convert to another format.

        :param targetFormat: Target format.
        :param targetDim: Target dimension.
        :type targetFormat: string
        :type targetFormat: integer

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
            if self.format_ == 'latlong':
                targetSize = (targetSize, 2*targetSize)
            if self.format_ == 'cube':
                targetSize = (targetSize, round(3/4*targetSize))

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
            self.data = self.data[:,:,np.newaxis]
        return self

    def setHemisphereAlbedo(self, normal, value):
        """Sets an whole hemisphere defined by `normal` to a given `value`
        (weighted by its solid angle).
        Useful to set the ground albedo."""
        raise NotImplementedError()

    def getMeanLightVectors(self, normals):
        """Compute the mean light vector of the environment map for the normals given.
        Normals should be 3xN.
        Output is 3xN.
        """
        normals = np.asarray(normals)
        solidAngles = self.solidAngles()
        solidAngles /= np.nansum(solidAngles) # Normalize to 1
        normals /= np.linalg.norm(normals, 1)

        x, y, z, _ = self.worldCoordinates()

        xyz = np.dstack((x, y, z))

        visibility = xyz.dot(normals) > 0

        intensity = deepcopy(self).toIntensity()
        meanlight = visibility * intensity.data * solidAngles[:,:,np.newaxis]
        meanlight = np.nansum(xyz[...,np.newaxis] * meanlight[...,np.newaxis].transpose((0,1,3,2)), (0, 1))

        return meanlight
