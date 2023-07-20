import numpy as np
import scipy, scipy.misc, scipy.ndimage, scipy.ndimage.filters
import scipy.spatial, scipy.interpolate, scipy.spatial.distance

from pysolar import solar

import envmap
from envmap.projections import latlong2world


def findBrightestSpot(image, minpct=99.99):
    """
    Find the sun position (in pixels, in the current projection) using the image.
    """
    if isinstance(image, envmap.EnvironmentMap):
        image = image.data

    # Gaussian filter
    filteredimg = scipy.ndimage.filters.gaussian_filter(image, (5, 5, 0))

    # Intensity image
    if filteredimg.ndim == 2 or filteredimg.shape[2] > 1:
        intensityimg = np.dot( filteredimg[:,:,:3], [.299, .587, .114] )
    else:
        intensityimg = filteredimg
    intensityimg[~np.isfinite(intensityimg)] = 0

    # Look for the value a the *minpct* percentage and threshold at this value
    # We do not take into account the pixels with a value of 0
    minval = np.percentile(intensityimg[intensityimg > 0], minpct)
    thresholdmap = intensityimg >= minval

    # Label the regions in the thresholded image
    labelarray, n = scipy.ndimage.measurements.label(thresholdmap, np.ones((3, 3), dtype="bool8"))

    # Find the size of each of them
    funcsize = lambda x: x.size
    patchsizes = scipy.ndimage.measurements.labeled_comprehension(intensityimg,
                                                                  labelarray,
                                                                  index=np.arange(1, n+1),
                                                                  func=funcsize,
                                                                  out_dtype=np.uint32,
                                                                  default=0.0)

    # Find the biggest one (we must add 1 because the label 0 is the background)
    biggestPatchIdx = np.argmax(patchsizes) + 1

    # Obtain the center of mass of the said biggest patch (we suppose that it is the sun)
    centerpos = scipy.ndimage.measurements.center_of_mass(intensityimg, labelarray, biggestPatchIdx)

    return centerpos


def sunPosition_fromEnvmap(envmapInput):
    """
    Finds the azimuth and zenith of the sun using the environnement map provided.
    Returns a tuple containing (zenith, azimuth)
    """
    c = findBrightestSpot(envmapInput.data)
    u, v = (c[1]+0.5) / envmapInput.data.shape[1], (c[0]+0.5) / envmapInput.data.shape[0]

    azimuth = np.pi*(2*u - 1)
    zenith = np.pi*v

    return zenith, azimuth


def sunPosition_pySolar_zenithAzimuth(latitude, longitude, time, elevation=0):
    """
    Finds the azimuth and zenith angle of the sun using the pySolar library.
    Takes latitude(deg), longitude(deg) and a datetime object.
    Returns a tuple containing (elevation, azimuth) in RADIANS with world coordinate orientation.

    Please note:
    zenith angle = 90degrees - elevation angle
    azimuth angle = north-based azimuth angles require offset (+90deg) and inversion (*-1) to measure clockwise
    thus, azimuth = (pi/2) - azimuth
    """
    
    # Find azimuth and elevation from pySolar library.
    azimuth = solar.get_azimuth(latitude, longitude, time, elevation)
    altitude = solar.get_altitude(latitude, longitude, time, elevation)

    # Convert to radians
    azimuth = (np.pi/2) + np.deg2rad(-azimuth)
    zenith = np.deg2rad(90 - altitude) 

    # Reset if degrees > 180
    if azimuth > np.pi: azimuth = azimuth - 2*np.pi
    if zenith > np.pi: zenith = zenith - 2*np.pi

    return zenith, azimuth


def sunPosition_pySolar_UV(latitude, longitude, time, elevation=0):
    """
    Finds the azimuth and elevation of the sun using the pySolar library.
    Takes latitude (in degrees), longitude(in degrees) and a datetime object.
    Returns a tuple containing the (x, y, z) world coordinate.

    Note, the validity (v) of the coordinate is not returned. 
    Please check the coordinate in respect to your environment map.
    """

    zenith, azimuth = sunPosition_pySolar_zenithAzimuth(
        latitude, longitude, 
        time, 
        elevation
    )
  
    # Fix orientation of azimuth
    azimuth = -(azimuth - (np.pi/2))
    
    # Convert to UV coordinates
    u = (azimuth/(2*np.pi))
    v = zenith/np.pi
    return u, v


def sunPosition_pySolar_XYZ(latitude, longitude, time, elevation=0):
    """
    Finds the azimuth and elevation of the sun using the pySolar library.
    Takes latitude (in degrees), longitude(in degrees) and a datetime object.
    Returns a tuple containing the (x, y, z) world coordinate.

    Note, the validity (v) of the coordinate is not returned. 
    Please check the coordinate in respect to your environment map.
    """

    u,v = sunPosition_pySolar_UV(
        latitude, longitude, 
        time, 
        elevation
    )
  
    # Convert to world coordinates
    x, y, z, _ = latlong2world(u, v)
    return x, y, z
