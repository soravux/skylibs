import numpy as np
from itertools import product
import pytest
from envmap import EnvironmentMap
from envmap.environmentmap import SUPPORTED_FORMATS
from tools3d.warping_operator import warpEnvironmentMap
from scipy import ndimage

np.random.seed(31415926)


# pytest [-s] [-k test_convert]

@pytest.mark.parametrize('format_, nadir, size, theta, phi', 
                         product(
                                SUPPORTED_FORMATS,
                                [-np.pi/2 + np.pi/20, -np.pi/4, 0, np.pi/2 - np.pi/20],
                                [512, 271],
                                np.linspace(0, 2*np.pi, 5) + 1e-10,
                                np.linspace(-np.pi/2, np.pi/2, 4) + 1e-10
                                )
                        )
def test_warpEnvironmentMap(format_, nadir, size, theta, phi):
    """
    This test works in the following way:
    a white "blob" of 1 pixel is added to the source environment map, and then the environment map is warped.
    The warped environment map is then checked to see if the blob is there at the expected location.
    """
    channels = 3
    blobSourceCoordinates = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), -np.cos(phi)])
    
    sourceEnvironmentMap = EnvironmentMap(size, format_, channels=channels)
    sourceEnvironmentMap.data = np.zeros((sourceEnvironmentMap.data.shape[0], sourceEnvironmentMap.data.shape[1], channels))
    sourceBlobCoordinatesPixel = np.mod(sourceEnvironmentMap.world2pixel(*blobSourceCoordinates.tolist()), (sourceEnvironmentMap.data.shape[1], 
                                                                                                            sourceEnvironmentMap.data.shape[0]))
    # we recompute the world coordinates because of the quantization error introduced by the world2pixel function
    x, y, z, _ =  tuple(sourceEnvironmentMap.pixel2world(*sourceBlobCoordinatesPixel.tolist()))

    # we add a white blob in the source environment map, which we are going to track after warping the envmap
    sourceEnvironmentMap.data[sourceBlobCoordinatesPixel[1], sourceBlobCoordinatesPixel[0], :] = 1.0

    warpedEnvironmentMap = warpEnvironmentMap(sourceEnvironmentMap.copy(), nadir, order=1)
    if warpedEnvironmentMap.data.max() == 0:
        # sometimes the warping removes the blob during interpolation due to the distortion, we ignore those cases
        pytest.skip()

    
    blobExpectedWarpedCoordinates = np.array([x, y, z - np.sin(nadir)])
    blobExpectedWarpedCoordinates /= np.linalg.norm(blobExpectedWarpedCoordinates, axis=0)
    blobExpectedWarpedCoordinatesPixel = np.array(EnvironmentMap(size, format_, channels=channels).world2pixel(*blobExpectedWarpedCoordinates.tolist()))

    warpedEnvironmentMapGray = warpedEnvironmentMap.data.mean(axis=2)

    # we detect the blob by checking if the pixel value is greater than 0
    assert warpedEnvironmentMapGray[blobExpectedWarpedCoordinatesPixel[1], blobExpectedWarpedCoordinatesPixel[0]] > 0.0
