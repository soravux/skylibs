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
                                [-np.pi/2 + np.pi/20, -np.pi/4, 0, np.pi/5, np.pi/2 - np.pi/20],
                                [512, 431, 271],
                                np.linspace(0, 2*np.pi, 5) + 1e-10,
                                np.linspace(-np.pi/2, np.pi/2, 6) + 1e-10
                                )
                        )
def test_warpEnvironmentMap(format_, nadir, size, theta, phi):
    blobSourceCoordinates = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), -np.cos(phi)])
    
    sourceEnvironmentMap = EnvironmentMap(size, format_, channels=3)
    sourceEnvironmentMap.data = np.zeros((sourceEnvironmentMap.data.shape[0], sourceEnvironmentMap.data.shape[1], 3))
    blobCoordinatesPixel = np.mod(sourceEnvironmentMap.world2pixel(*blobSourceCoordinates.tolist()), (sourceEnvironmentMap.data.shape[1], 
                                                                                                  sourceEnvironmentMap.data.shape[0]))
    blobSourceCoordinates = sourceEnvironmentMap.pixel2world(*blobCoordinatesPixel.tolist())
    # we recompute the world coordinates because of the quantization error introduced by the world2pixel function
    x, y, z, _ = tuple(blobSourceCoordinates)

    # we add a white blob in the source environment map, which we are going to track after warping the envmap
    sourceEnvironmentMap.data[blobCoordinatesPixel[1], blobCoordinatesPixel[0], :] = 1.0

    warpedEnvironmentMap = warpEnvironmentMap(sourceEnvironmentMap.copy(), nadir, order=1)
    if warpedEnvironmentMap.data.max() == 0:
        pytest.skip()
    
    warpedEnvironmentMapGray = warpedEnvironmentMap.data.mean(axis=2)
    # dilate the blob
    blobActualCoordinatesPixel = np.array(np.unravel_index(np.argmax(warpedEnvironmentMapGray), warpedEnvironmentMapGray.shape))[[1, 0]]

    blobExpectedWarpedCoordinates = np.array([x, y, z - np.sin(nadir)])
    blobExpectedWarpedCoordinates /= np.linalg.norm(blobExpectedWarpedCoordinates, axis=0)
    blobExpectedWarpedCoordinatesPixel = np.array(EnvironmentMap(size, format_, channels=3).world2pixel(*blobExpectedWarpedCoordinates.tolist()))
    
    print(f'blobActualCoordinatesPixel: {blobActualCoordinatesPixel}')
    print(f'blobExpectedWarpedCoordinatesPixel: {blobExpectedWarpedCoordinatesPixel}')
    print(f'size: {warpedEnvironmentMapGray.shape}')

    warpedEnvironmentMapGray = ndimage.maximum_filter(warpedEnvironmentMapGray, size=3)

    # we detect the blob by checking if the pixel value is greater than 0
    assert warpedEnvironmentMapGray[blobExpectedWarpedCoordinatesPixel[1], blobExpectedWarpedCoordinatesPixel[0]] > 0.0
