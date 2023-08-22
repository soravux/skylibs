import envmap
import numpy as np

cachedWorldCoordinates = {}
def warpEnvironmentMap(environmentMap, nadir, order=1):
    """
    Applies a warping operation to the environment map by simulating a camera translation along the z-axis
    (the environment map is approximated by a sphere, thus occlusions are not taken into account).
    The translation amount is determined by the sinus of the nadir angle.

    :param envmap: Environment map to warp.
    :param nadir: Nadir angle in radians.
    :param order: Interpolation order (0: nearest, 1: linear, ..., 5).

    This code is a refactored version of the orignal implementation by Marc-André Gardner.
    The algorithm is described in the following paper:
    “Learning to predict indoor illumination from a single image | ACM Transactions on Graphics.”
    https://dl.acm.org/doi/10.1145/3130800.3130891.
    """
    assert isinstance(environmentMap, envmap.EnvironmentMap)

    global cachedWorldCoordinates
    cacheKey = (environmentMap.data.shape, environmentMap.format_)
    if not cacheKey in cachedWorldCoordinates:
        cachedWorldCoordinates[cacheKey] = environmentMap.worldCoordinates()
    
    def warpCoordinates(x, y, z, zOffset):
        """
        Moves the x, y, z coordinates (ray intersections with the unit sphere,
        assuming the origin is at [0, 0, 0]) to their new position, where the ray origins
        are at [0, 0, zOffset] and the ray directions are unchanged.

        The equation for the new coordinates is a simplified version the quadratic
        formula in the eq. 3 in the paper where we know v_x^2+v_y^2+v_z^2 = 1.
        We only keep the positive solution since the negative one would move the
        point to the other side of the sphere.
        """
        t = -z * zOffset + np.sqrt(zOffset**2 * (z**2 - 1) + 1)
        return x * t, y * t, z * t + zOffset

    xDestination, yDestination, zDestination, _ = cachedWorldCoordinates[cacheKey]
    xSource, ySource, zSource = warpCoordinates(xDestination, yDestination, zDestination, np.sin(nadir))
    uSource, vSource = environmentMap.world2image(xSource, ySource, zSource)
    environmentMap.interpolate(uSource, vSource, order=order)

    return environmentMap

