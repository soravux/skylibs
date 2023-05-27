import os
import pathlib
from copy import deepcopy
from decimal import Decimal

from tqdm import tqdm
import numpy as np
from scipy.ndimage import map_coordinates, zoom
from skimage.transform import resize_local_mean, downscale_local_mean

from hdrio import imread
from .tetrahedronSolidAngle import tetrahedronSolidAngle
from .projections import *
from .xmlhelper import EnvmapXMLParser
from .rotations import rotx, roty, rotz


SUPPORTED_FORMATS = [
    'angular',
    'skyangular',
    'latlong',
    'skylatlong',
    'sphere',
    'cube',
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


def isPath(var):
    return isinstance(var, str) or isinstance(var, pathlib.PurePath)


class EnvironmentMap:
    def __init__(self, im, format_=None, copy=True, channels=3):
        """
        Creates an EnvironmentMap.

        :param im: Image path (str, pathlib.Path) or data (np.ndarray) representing 
                   an EnvironmentMap, or the height (int) of an empty EnvironmentMap.
        :param format_: EnvironmentMap format. Can be any from SUPPORTED_FORMATS.
        :param copy: When a numpy array is given, should it be copied.
        :param channels: Number of channels (e.g., 1=grayscale, 3=color).
        :type im: str, Path, int, np.ndarray
        :type format_: str
        :type copy: bool
        """
        if not format_ and isPath(im):
            filename = os.path.splitext(str(im))[0]
            metadata = EnvmapXMLParser("{}.meta.xml".format(filename))
            format_ = metadata.getFormat()

        assert format_ is not None, (
            "Please provide format (metadata file not found).")
        assert format_.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(format_))

        self.format_ = format_.lower()

        if isPath(im):
            # We received the filename
            self.data = imread(str(im))
        elif isinstance(im, int):
            # We received a single scalar
            if self.format_ == 'latlong':
                self.data = np.zeros((im, im*2, channels))
            elif self.format_ == 'skylatlong':
                self.data = np.zeros((im, im*4, channels))
            elif self.format_ == 'cube':
                self.data = np.zeros((im, round(3/4*im), channels))
            else:
                self.data = np.zeros((im, im, channels))
        elif type(im).__module__ == np.__name__:
            # We received a numpy array
            self.data = np.asarray(im, dtype='double')

            if copy:
                self.data = self.data.copy()
        else:
            raise Exception('Could not understand input. Please provide a '
                            'filename (str), an height (integer) or an image '
                            '(np.ndarray).')

        self.backgroundColor = np.zeros(self.data.shape[-1])
        self.validate()

    def validate(self):
        # Ensure the envmap is valid
        if self.format_ in ['sphere', 'angular', 'skysphere', 'skyangular']:
            assert self.data.shape[0] == self.data.shape[1], (
                "Sphere/Angular formats must have the same width/height")
        elif self.format_ == 'latlong':
            assert 2*self.data.shape[0] == self.data.shape[1], (
                "LatLong format width should be twice the height")
        elif self.format_ == 'skylatlong':
            assert 4*self.data.shape[0] == self.data.shape[1], (
                "SkyLatLong format width should be four times the height")

        assert self.data.ndim == 3, "Expected 3-dim array. For grayscale, use [h,w,1]."

    @classmethod
    def fromSkybox(cls, top, bottom, left, right, front, back):
        """Create an environment map from skybox (cube) captures. Six images
        must be provided, one for each side of the virtual cube. This function
        will return an EnvironmentMap object.

        All the images must be square (width==height), have the same size 
        and number of channels.
        """
        basedim = top.shape[0]
        channels = 1 if len(top.shape) == 2 else top.shape[2]

        cube = np.zeros((basedim*4, basedim*3, channels), dtype=top.dtype)
        cube[0:basedim, basedim:2*basedim] = top
        cube[basedim:2*basedim, basedim:2*basedim] = front
        cube[basedim:2*basedim, 2*basedim:3*basedim] = right
        cube[3*basedim:4*basedim, basedim:2*basedim] = np.fliplr(np.flipud(back))
        cube[1*basedim:2*basedim, 0:basedim] = left
        cube[2*basedim:3*basedim, basedim:2*basedim] = bottom

        # We ensure that there is no visible artifacts at the junction of the images
        # due to the interpolation process
        cube[0:basedim, basedim-1] = left[0,...] # top-left
        cube[0:basedim, 2*basedim] = right[0,...][::-1] # top-right
        cube[basedim-1, 2*basedim:3*basedim] = top[:,-1][::-1] #right-top
        cube[2*basedim, 2*basedim:3*basedim] = bottom[:,-1] #right-bottom
        cube[1*basedim-1, 0:basedim] = top[:,0] # left-top
        cube[2*basedim, 0:basedim] = bottom[:,0][::-1] # left-bottom
        cube[2*basedim:3*basedim, basedim-1] = left[-1,...][::-1] #bottom-left
        cube[2*basedim:3*basedim, 2*basedim] = right[-1,...] #bottom-right
        cube[3*basedim:, basedim-1] = left[:,0][::-1] # back-left
        cube[3*basedim:, 2*basedim] = right[:,-1][::-1] # back-right

        return cls(cube, format_="cube")


    @classmethod
    def from_omnicam(cls, im, targetDim, targetFormat='skyangular', OcamCalib_=None, copy=True, order=1):
        """
        Creates an EnvironmentMap from Omnidirectional Camera (OcamCalib) capture.

        :param im: Image path (str, pathlib.Path) or data (np.ndarray) representing 
                an ocam image.
        :param OcamCalib: OCamCalib calibration dictionary. If not provided and 
                param im is an image path, then calibration will be loaded directly 
                from matching ".meta.xml" file.
        :param targetFormat: Target format.
        :param targetDim: Target dimension.
        :param order: Interpolation order (0: nearest, 1: linear, ..., 5).
        :param copy: When a numpy array is given, should it be copied.

        :type im: str, Path, np.ndarray
        :type OcamCalib: dict
        :type targetFormat: string
        :type targetDim: integer
        :type order: integer (0,1,...,5)
        :type copy: bool
        """

        if not OcamCalib_ and isPath(im):
            filename = os.path.splitext(str(im))[0]
            metadata = EnvmapXMLParser("{}.meta.xml".format(filename))
            OcamCalib_ = metadata.get_calibration()

        assert OcamCalib_ is not None, (
            "Please provide OCam (metadata file not found).")

        if isPath(im):
            # We received the filename
            data = imread(str(im))
        elif type(im).__module__ == np.__name__:
            # We received a numpy array
            data = np.asarray(im, dtype='double')
            if copy:
                data = data.copy()
        else:
            raise Exception('Could not understand input. Please provide a '
                            'filename (str) or an image (np.ndarray).')

        e = EnvironmentMap(targetDim, targetFormat)
        dx, dy, dz, valid = e.worldCoordinates()
        u,v = world2ocam(dx, dy, dz, OcamCalib_)

        # Interpolate
        # Repeat the first and last rows/columns for interpolation purposes
        h, w, d = data.shape
        source = np.empty((h + 2, w + 2, d))

        source[1:-1, 1:-1] = data
        source[0,1:-1] = data[0,:]; source[0,0] = data[0,0]; source[0,-1] = data[0,-1]
        source[-1,1:-1] = data[-1,:]; source[-1,0] = data[-1,0]; source[-1,-1] = data[-1,-1]
        source[1:-1,0] = data[:,0]
        source[1:-1,-1] = data[:,-1]

        # To avoid displacement due to the padding
        u += 0.5/data.shape[1]
        v += 0.5/data.shape[0]
        target = np.vstack((u.flatten()*data.shape[0], v.flatten()*data.shape[1]))

        data = np.zeros((u.shape[0], u.shape[1], d))
        for c in range(d):
            map_coordinates(source[:,:,c], target, output=data[:,:,c].reshape(-1), cval=np.nan, order=order, prefilter=filter)
        e.data = data

        # Done
        return e


    def __hash__(self):
        """Provide a hash of the environment map type and size.
        Warning: doesn't take into account the data, just the type,
                 and the size (useful for soldAngles)."""
        return hash((self.data.shape, self.format_))

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
        omega += tetrahedronSolidAngle(b, c, d)

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
            'skylatlong': skylatlong2world,
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
            'skylatlong': world2skylatlong,
            'sphere': world2sphere,
            'cube': world2cube,
        }.get(self.format_)
        return func(x, y, z)


    def world2pixel(self, x, y, z):
        """Returns the (u, v) coordinates (in the interval defined by the MxN image)."""

        # Get (u,v) in [0, 1] interval
        u,v = self.world2image(x, y, z)

        # de-Normalize coordinates to interval defined by the MxN image
        u = np.floor(u*self.data.shape[1]).astype(int)
        v = np.floor(v*self.data.shape[0]).astype(int)

        return u, v


    def pixel2world(self, u, v):
        """Returns the (x, y, z) coordinates for pixel cordinates (u,v)(in the interval defined by the MxN image)."""

        # Normalize coordinates to [0, 1] interval
        u = (u+0.5) / self.data.shape[1]
        v = (v+0.5) / self.data.shape[0]

        # Get (x, y, z) 
        x, y, z, v = self.image2world(u, v)

        return x, y, z, v

    def interpolate(self, u, v, valid=None, order=1, filter=True):
        """
        Interpolate to get the desired pixel values.

        :param order: Interpolation order (0: nearest, 1: linear, ..., 5).
        :type order: integer (0,1,...,5)
        """

        u = u.copy()
        v = v.copy()

        # Repeat the first and last rows/columns for interpolation purposes
        h, w, d = self.data.shape
        source = np.empty((h + 2, w + 2, d))

        source[1:-1, 1:-1] = self.data
        source[0,1:-1] = self.data[0,:]; source[0,0] = self.data[0,0]; source[0,-1] = self.data[0,-1]
        source[-1,1:-1] = self.data[-1,:]; source[-1,0] = self.data[-1,0]; source[-1,-1] = self.data[-1,-1]
        source[1:-1,0] = self.data[:,0]
        source[1:-1,-1] = self.data[:,-1]

        # To avoid displacement due to the padding
        u += 0.5/self.data.shape[1]
        v += 0.5/self.data.shape[0]
        target = np.vstack((v.flatten()*self.data.shape[0], u.flatten()*self.data.shape[1]))

        data = np.zeros((u.shape[0], u.shape[1], d))
        for c in range(d):
            map_coordinates(source[:,:,c], target, output=data[:,:,c].reshape(-1), cval=np.nan, order=order, prefilter=filter)
        self.data = data

        if valid is not None:
            self.setBackgroundColor(self.backgroundColor, valid)

        return self

    def setBackgroundColor(self, color, valid=None):
        """Sets the area defined by ~valid to color."""
        if valid is None:
            _, _, _, valid = self.worldCoordinates()

        assert valid.dtype == 'bool', "`valid` must be a boolean array."
        assert valid.shape[:2] == self.data.shape[:2], "`valid` must be the same size as the EnvironmentMap."

        self.backgroundColor = np.asarray(color)
        if self.backgroundColor.size == 1 and self.data.shape[2] != self.backgroundColor.size:
            self.backgroundColor = np.tile(self.backgroundColor, (self.data.shape[2],))

        assert self.data.shape[2] == self.backgroundColor.size, "Channel number mismatch when setting background color"

        mask = np.invert(valid)
        if mask.sum() > 0:
            self.data[np.tile(mask[:,:,None], (1, 1, self.data.shape[2]))] = np.tile(self.backgroundColor, (mask.sum(),))

        return self

    def convertTo(self, targetFormat, targetDim=None, order=1):
        """
        Convert to another format.

        :param targetFormat: Target format.
        :param targetDim: Target dimension.
        :param order: Interpolation order (0: nearest, 1: linear, ..., 5).

        :type targetFormat: string
        :type targetDim: integer
        :type order: integer (0,1,...,5)
        """
        self.validate()
        assert targetFormat.lower() in SUPPORTED_FORMATS, (
            "Unknown format: {}".format(targetFormat))
        assert order in range(6), "Spline interpolation order must be between 0 and 5."

        if not targetDim:
            # By default, number of rows
            targetDim = self.data.shape[0]

        eTmp = EnvironmentMap(targetDim, targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        u, v = self.world2image(dx, dy, dz)
        self.format_ = targetFormat.lower()
        self.interpolate(u, v, valid, order)

        return self

    def rotate(self, dcm, order=1):
        """
        Rotate the environment map.

        :param dcm: Rotation information (currently only 3x3 numpy matrix)
        :param order: Integer interpolation order (0: nearest, 1: linear, ..., 5).

        """

        self.validate()
        assert type(dcm).__module__ == np.__name__ and dcm.ndim == 2 and dcm.shape == (3, 3)
        assert order in range(6), "Spline interpolation order must be between 0 and 5."

        dx, dy, dz, valid = self.worldCoordinates()

        ptR = np.dot(dcm, np.vstack((dx.flatten(), dy.flatten(), dz.flatten())))
        dx, dy, dz = ptR[0].reshape(dx.shape), ptR[1].reshape(dy.shape), ptR[2].reshape(dz.shape)

        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)
        dz = np.clip(dz, -1, 1)

        u, v = self.world2image(dx, dy, dz)
        self.interpolate(u, v, valid, order)

        return self

    def resize(self, targetSize, order=1, debug=False):
        """
        Resize the current environnement map to targetSize.
        The energy-preserving "pixel mixing" algorithm is used when downscaling unless 
        order is set to 0 (nearest neighbor interpolation).

        `targetSize` is either the desired `height` or `(height, width)`.
        `order` is the order of the spline interpolation (0: nearest, 1: linear, ..., 5).

        """

        self.validate()
        assert order in range(6), "Spline interpolation order must be between 0 and 5."

        if not isinstance(targetSize, tuple):
            if self.format_ == 'latlong':
                targetSize = (targetSize, 2*targetSize)
            elif self.format_ == 'skylatlong':
                targetSize = (targetSize, 4*targetSize)
            elif self.format_ == 'cube':
                targetSize = (targetSize, round(3/4*targetSize))
            else:
                targetSize = (targetSize, targetSize)
        
        if debug == True:
            old_mean = self.data.mean()

        # downsampling
        if targetSize[0] < self.data.shape[0] and order != 0:

            # check if integer
            if (Decimal(self.data.shape[0]) / Decimal(targetSize[0])) % 1 == 0:
                if debug is True:
                    print("integer resize")
                fac = self.data.shape[0] // targetSize[0]
                self.data = downscale_local_mean(self.data, (fac, fac, 1))
            else:    
                if debug is True:
                    print("non-integer resize")
                self.data = resize_local_mean(self.data, targetSize, grid_mode=True, preserve_range=True)

        else: # upsampling or nearest neighbor
            _size = []
            for i in range(2):
                _size.append(targetSize[i] / self.data.shape[i] if targetSize[i] > 1. else targetSize[i])
            _size.append(1.0)
            self.data = zoom(self.data, _size, order=order)

        if debug is True:
            print("Energy difference in resize: {:.04f}".format(self.data.mean()/old_mean - 1.))

        return self

    def toIntensity(self, mode="ITU BT.709", colorspace="linear"):
        """
        Converts the environment map to grayscale.

        `mode` can be either :
               "ITU BT.601" (luma in sRGB),
               "ITU BT.709" (assumes the envmap is linear),
               mean (mean of the channels)
        `colorspace`: either "sRGB" or "linear"
        """
        self.validate()

        if self.data.shape[2] != 3:
            print("Envmap does not have 3 channels. This function won't do anything.")
            return self

        if colorspace.lower() == "sRGB":
            if self.data.max() > 1.:
                raise Exception("Error during sRGB to linear conversion: data is > 1. Please linearize "
                                "the data beforehand or normalize it [0, 1].")
            self.data = self.data**2.2

        if mode == "ITU BT.601":
            self.data = 0.299*self.data[...,0] + 0.587*self.data[...,1] + 0.114*self.data[...,2]
            self.data = self.data[:,:,np.newaxis]
        elif mode == "ITU BT.709":
            self.data = 0.2126*self.data[...,0] + 0.7152*self.data[...,1] + 0.0722*self.data[...,2]
            self.data = self.data[:,:,np.newaxis]
        elif mode == "mean":
            self.data = np.mean(self.data, axis=2, keepdims=True)

        if colorspace.lower() == "sRGB":
            self.data = self.data**(1./2.2)

        return self

    def getHemisphere(self, normal, channels=True):
        """
        normal: 
        """
        normal = np.asarray(normal, dtype=np.float32).reshape((-1))
        assert normal.size == 3, "unknown normal shape, should have 3 elements"
        normal /= np.linalg.norm(normal)

        x, y, z, _ = self.worldCoordinates()
        xyz = np.dstack((x, y, z))

        mask = xyz.dot(normal) > 0
        if channels == False:
            return mask

        return np.tile(mask[:,:,None], (1, 1, self.data.shape[2]))

    def setHemisphereValue(self, normal, value):
        """Sets an whole hemisphere defined by `normal` to a given `value`."""

        mask = self.getHemisphere(normal, channels=True)

        value = np.asarray(value)
        assert value.size in (self.data.shape[2], 1), ("Cannot understand value: should "
            "be size 1 or channels ({})".format(self.data.shape[2]))

        if value.size == 1 and self.data.shape[2] != value.size:
            value = np.tile(value, (self.data.shape[2],))

        self.data[mask] = np.tile(value, (mask.sum()//self.data.shape[2],))

        return self

    def getMeanLightVectors(self, normals):
        """Compute the mean light vector of the environment map for the normals given.
        Normals should be 3xN.
        Output is 3xN.
        """
        self.validate()

        normals = np.asarray(normals)
        normals /= np.linalg.norm(normals, 1)
        solidAngles = self.solidAngles()
        solidAngles /= np.nansum(solidAngles) # Normalize to 1

        x, y, z, _ = self.worldCoordinates()
        xyz = np.dstack((x, y, z))

        visibility = xyz.dot(normals) > 0

        intensity = deepcopy(self).toIntensity()
        meanlight = visibility * intensity.data * solidAngles[:,:,np.newaxis]
        meanlight = np.nansum(xyz[...,np.newaxis] * meanlight[...,np.newaxis].transpose((0,1,3,2)), (0, 1))

        return meanlight

    def blur(self, kappa):
        """
        Blurs the environment map, taking into account the format.
        Uses the von Mises-Fisher notation (`kappa` >= 0 for blur bandwidth, higher values are sharper)
        """

        x, y, z, _ = self.worldCoordinates()
        xyz = np.dstack((x, y, z)).reshape((-1, 3))

        h, w, c = self.data.shape[:3]
        data = self.data.reshape((-1, c))
        sa = self.solidAngles().reshape((-1, 1))

        C3 = kappa/(4*np.pi*np.sinh(kappa))

        result = np.empty((h*w, c), dtype=np.float32)

        for i in tqdm(range(xyz.shape[0])):
            normal = xyz[i:i+1,:]
            result[i,:] = np.sum(data*sa*C3*np.exp(kappa*np.sum(xyz*normal, 1, keepdims=True)), 0)

        self.data = result.reshape((h, w, c))
        return self

    def embed(self, vfov, rotation_matrix, image, order=1):
        """
        Projects an image onto the environment map.

        :vfov: Vertical Field of View (degrees).
        :rotation_matrix: Camera rotation matrix.
        :image: The image to project.
        :param order: Integer interpolation order (0: nearest, 1: linear, ..., 5).
        """

        self.validate()
        assert order in range(6), "Spline interpolation order must be between 0 and 5."

        targetDim = self.data.shape[0]
        targetFormat = self.format_

        eTmp = EnvironmentMap(targetDim, targetFormat)
        dx, dy, dz, valid = eTmp.worldCoordinates()
        ar = image.shape[1]/image.shape[0]

        vfov_half_rad = vfov/2.*np.pi/180.
        hfov_half_rad = np.arctan(np.tan(vfov_half_rad)*ar)

        fx = 0.5/np.tan(hfov_half_rad)
        fy = 0.5/np.tan(vfov_half_rad)
        u0 = 0.5
        v0 = 0.5

        # world2image for a camera
        K = np.array([[fx, 0, u0],
                      [0, fy, v0],
                      [0,  0,  1]])
        M = K.dot(rotation_matrix)

        xyz = np.dstack((dx, dy, dz)).reshape((-1, 3)).T

        # mask behind the camera
        forward_vector = rotation_matrix.T.dot(np.array([0, 0, -1]).T)
        mask = forward_vector.dot(xyz) <= 0
        xyz[:,mask] = np.inf

        uv = M.dot(xyz)
        u = (uv[0,:]/uv[2,:]).reshape(self.data.shape[:2])
        v = (uv[1,:]/uv[2,:]).reshape(self.data.shape[:2])

        self.format_ = targetFormat.lower()
        self.data = image.copy()[:,::-1,...]
        self.interpolate(u, v, valid, order)
        return self

    def project(self, vfov, rotation_matrix, ar=4./3., resolution=(640, 480),
                projection="perspective", mode="normal", order=1):
        """
        Perform a projection onto a plane ("_simulates_" a camera).

        Note: this function does not modify the foreshortening present in the
        environment map.

        :vfov: Vertical Field of View (degrees).
        :rotation_matrix: Camera rotation matrix.
        :ar: Aspect ratio (width / height), defaults to 4/3.
        :resolution: Output size in (cols, rows), defaults to (640,480).
        :projection: perspective (default) or orthographic.
        :mode: "normal": perform crop (default)
               "mask": show pixel mask in the envmap,
               "normal+uv": returns (crop, u, v), where (u,v) are the coordinates
                            of the crop.
        :param order: Integer interpolation order (0: nearest, 1: linear, ..., 5).
        """

        self.validate()
        assert order in range(6), "Spline interpolation order must be between 0 and 5."

        coords = self._cameraCoordinates(vfov, rotation_matrix, ar,
                                         resolution, projection, mode)

        if mode == "mask":
            return coords

        target = self.copy()
        if target.format_ != "latlong":
            target = target.convertTo("LatLong")

        # Get image coordinates from world sphere coordinates
        u, v = target.world2image(coords[0,:], coords[1,:], coords[2,:])
        u, v = u.reshape(resolution[::-1]), v.reshape(resolution[::-1])

        crop = target.interpolate(u, v, order=order).data

        if mode == "normal+uv":
            return crop, u, v

        return crop

    def _cameraCoordinates(self, vfov, rotation_matrix, ar=4./3., resolution=(640, 480),
            projection="perspective", mode="normal"):

        if mode not in ("normal", "mask", "normal+uv"):
            raise Exception("Unknown mode: {}.".format(mode))

        if projection == "orthographic":
            vfov = np.arctan(np.sin(vfov*np.pi/180.))*180/np.pi

        # aspect ratio in pixels
        hfov = 2 * np.arctan(np.tan(vfov*np.pi/180./2)*ar)*180/np.pi

        # Project angle on the sphere to the +Z plane (distance=1 from the camera)
        mu = np.tan(hfov/2.*np.pi/180.)
        mv = np.tan(vfov/2.*np.pi/180.)

        if mode == "mask":
            x, y, z, _ = self.worldCoordinates()
            xy = np.sqrt( (x**2 + y**2) / (-x**2 - y**2 + 1) )
            theta = np.arctan2(x, y)
            x = xy*np.sin(theta)
            y = xy*np.cos(theta)

            hmask = (x > -mu) & (x < mu)
            vmask = (y > -mv) & (y < mv)
            dmask = z < 0
            mask = hmask & vmask & dmask

            e = EnvironmentMap(mask[:,:,np.newaxis], self.format_)
            e.rotate(rotation_matrix)

            mask = e.data[:,:,0]
            return mask

        # Uniform sampling on the plane
        dy = np.linspace(mv, -mv, resolution[1])
        dx = np.linspace(-mu, mu, resolution[0])
        x, y = np.meshgrid(dx, dy)

        # Compute unit length vector (project back to sphere) from the plane
        x, y = x.ravel(), y.ravel()
        if projection == "perspective":
            xy = np.sqrt( (x**2 + y**2) / (x**2 + y**2 + 1) )
            theta = np.arctan2(x, y)
            x = xy*np.sin(theta)
            y = xy*np.cos(theta)
        elif projection == "orthographic":
            pass
        else:
            raise Exception("Unknown projection: {}.".format(projection))
        z = -np.sqrt(1 - (x**2 + y**2))
        coords = np.vstack((x, y, z))

        # Perform rotation
        coords = rotation_matrix.T.dot(coords)

        return coords


def rotation_matrix(azimuth, elevation, roll=0):
    """Returns a camera rotation matrix.
    :azimuth: left (negative) to right (positive) [rad]
    :elevation: upward (negative) to downward (positive) [rad]
    :roll: counter-clockwise (negative) to clockwise (positive) [rad]"""
    return rotz(roll).dot(rotx(elevation)).dot(roty(-azimuth))


def downscaleEnvmap(nenvmap, sao, sat, times):
    """Deprecated"""
    print("This function is deprecated and will be removed in an ulterior version of skylibs.")
    sz = sat.shape[0]
    return nenvmap.resize(sz)
