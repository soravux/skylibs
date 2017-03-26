import numpy as np
from scipy.special import sph_harm
from tqdm import tqdm

from envmap import EnvironmentMap

# Sanity check, sph_harm was bogus in some versions of scipy / Anaconda Python
# http://stackoverflow.com/questions/33149777/scipy-spherical-harmonics-imaginary-part
assert(np.isclose(sph_harm(2, 5, 2.1, 0.4), -0.17931012976432356-0.31877392205957022j))


# References:
# https://www.cs.dartmouth.edu/~wjarosz/publications/dissertation/appendixB.pdf
# http://www.ppsloan.org/publications/StupidSH36.pdf
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
# http://www.graphics.cornell.edu/pubs/1991/SAWG91.pdf


def sphericalHarmonicsTransform(envmap, degrees=None, reduction_type='right'):
    """
    Performs a Spherical Harmonics Transform of `envmap` up to `degree`.
    """
    ch = envmap.data.shape[2] if len(envmap.data.shape) > 2 else 1

    if degrees is None:
        degrees = np.ceil(np.maximum(envmap.shape) / 2.)

    retval = np.zeros(((degrees + 1)**2, ch), dtype=np.complex)

    f = envmap.data * envmap.solidAngles()[:,:,np.newaxis]

    x, y, z, valid = envmap.worldCoordinates()
    theta = np.arctan2(x, -z)
    phi = np.arccos(y)

    for l in tqdm(range(degrees + 1)):
        for col, m in enumerate(range(-l, l + 1)):
            Y = sph_harm(m, l, theta, phi)
            for c in range(ch):
                retval[l**2+col,c] = np.nansum(Y*f[:,:,c])
    return removeRedundantCoeffs(retval, reduction_type)


def inverseSphericalHarmonicsTransform(coeffs, envmap_height=512, format_='latlong', reduction_type='right'):
    """
    Recovers an EnvironmentMap from a list of coefficients.
    """
    degrees = np.asscalar(np.sqrt(coeffs.shape[0]).astype('int')) - 1

    coeffs = addRedundantCoeffs(coeffs, reduction_type)[..., np.newaxis]

    ch = coeffs.shape[1] if len(coeffs.shape) > 0 else 1
    retval = EnvironmentMap(envmap_height, format_)
    retval.data = np.zeros((retval.data.shape[0], retval.data.shape[1], ch), dtype=np.float32)

    x, y, z, valid = retval.worldCoordinates()
    theta = np.arctan2(x, -z)
    phi = np.arccos(y)

    for l in range(degrees + 1):
        for col, m in enumerate(range(-l, l + 1)):
            Y = sph_harm(m, l, theta, phi)
            for c in range(ch):
                retval.data[..., c] += (coeffs[l**2+col, c]*Y).real

    return retval


def _triangleRightSide(x):
    # see A004201, Boris Putievskiy, Dec 13 2012
    # and A000217, Paul Barry, May 29 2006
    return np.asarray([(n + np.floor((-1+np.sqrt(8*n-7))/2)*(np.floor((-1+np.sqrt(8*n-7))/2)+1)/2).astype('int')
                        for n in range(1, int((2*(x + 1)+1)**2/8) + 1)]) - 1


def removeRedundantCoeffs(coeffs, reduction_type):

    degrees = np.asscalar(np.sqrt(coeffs.shape[0]).astype('int')) - 1

    if reduction_type is None:
        return coeffs
    elif reduction_type == 'imag_real':
        raise Exception('NotImplementedYet')
    elif reduction_type == 'right':
        return coeffs[_triangleRightSide(degrees)]

    raise Exception('unknown reduction_type')


def addRedundantCoeffs(coeffs, reduction_type):

    degrees = round(np.asscalar(1./2. * (np.sqrt(8*coeffs.shape[0]+1) - 1))) - 1
    retval = np.empty(((degrees + 1)**2, *coeffs.shape[1:]), dtype=np.complex128)

    if reduction_type is None:
        return coeffs
    elif reduction_type == 'image_real':
        raise Exception('NotImplementedYet')
    elif reduction_type == 'right':
        retval[_triangleRightSide(degrees)] = coeffs
        print(set(range((degrees + 1)**2)) - set(_triangleRightSide(degrees)))
        for i in set(range((degrees + 1)**2)) - set(_triangleRightSide(degrees)):
            l = np.sqrt(i).astype('int')
            m = abs((l) - (i - l**2))
            print(l, m)
            retval[i] = (-1)**m * np.conj(retval[l**2 + l + m])
        return retval

    raise Exception('unknown reduction_type')


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    e = EnvironmentMap('envmap.exr', 'angular')
    e.resize((256, 256))
    e.convertTo('latlong')
    coeffs = sphericalHarmonicsTransform(e, 4)
    er = inverseSphericalHarmonicsTransform(coeffs, 256)
    plt.subplot(1,2,1); plt.imshow(e.data)
    plt.subplot(1,2,2); plt.imshow(er.data)
    plt.show()