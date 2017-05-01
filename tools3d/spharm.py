import functools
import os

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
# A Fast Transform for Spherical Harmonics, Martin J. Mohlenkamp, 1999 -> Lower complexity, but unstable at high pixel count
# Efficient Spherical Harmonic Transforms aimed at pseudo-spectral numerical simulations, Nathanael Schaeffer, 2015
# https://github.com/rlk/sht/
# http://www.sciencedirect.com/science/article/pii/S0098300405001780



def FSHT(envmap, degrees=3, reduction_type='right'):
    """
    Fast Spherical Harmonic Transform.
    Always outputs with reduction_type='right'
    """
    if reduction_type != 'right':
        raise NotImplemented()


    from cffi import FFI
    ffi = FFI()
    ffi.cdef("""
        void generateAssociatedLegendreFactors(const float N, float *data_out, const float * nodes, const unsigned int num_nodes);
    """)
    C = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)), "spharm_tools.dll"))

    envmap.data = envmap.data[...,np.newaxis]
    ch = envmap.data.shape[2]

    # x, y, z, valid = envmap.worldCoordinates()
    # theta = np.arctan2(x, -z)
    # phi = np.arccos(y).ravel().copy()

    f = np.atleast_3d(envmap.data)

    fm = np.fft.fft(envmap.data, axis=1)#, norm='ortho')

    # nodes = []
    # weights = []
    # for d in range(1, degrees + 2):
    #     x, y = np.polynomial.legendre.leggauss(d)
    #     nodes.extend(x)
    #     weights.extend(y)

    retval = np.zeros((int((2*(degrees + 1)+1)**2/8), ch), dtype=np.complex128)

    P, nodes = _getP(envmap, degrees)
    print(degrees, P.shape)

    # Gauss-Legendre / Gauss-Chebyshev quadrature to speed up?
    # Perform in C
    # for j in range(envmap.data.shape[1]):
    #     for k in range(ch):
    #         #fmi = np.interp(nodes, np.linspace(-np.pi/2, np.pi/2, fm.shape[0]), np.squeeze(fm[:,j,k]))
    #         i = 0
    #         for l in range(degrees + 1):
    #             for m in range(0, l + 1):
    #                 retval[l,m,k] += fm * P[l,m]
    #                 i += 1

    import operator
    i = 0
    for l in tqdm(range(degrees + 1)):
        for m in range(0, l + 1):
            #coef = np.sqrt(( (2.*l+1.) / (4.*np.pi) ) * ( 1. / (functools.reduce(operator.mul, range(l-m+1, l+m+1), 1)) ) )
            for c in range(ch):
                #retval[i,c] = np.nansum(coef*P[:,i]*np.squeeze(fm[:,m,c]))
                retval[i,c] = np.nansum(P[:,i]*np.squeeze(fm[:,m,c]))
                #import pdb; pdb.set_trace()
                #retval[i,c] = np.nansum(coef*P[:,i]*np.exp(1j*m*theta.ravel())*envmap.data.reshape([-1,ch])[:,c])
                #retval[i,c] = np.nansum(coef*ref[:,i]*np.exp(-1j*m*(theta).ravel())*f[:,c])
            #print("2: ", coef*P[:,i]*np.exp(1j*m*theta.ravel())*f[:,c])
            i += 1

    #import pdb; pdb.set_trace()

    from matplotlib import pyplot as plt
    #plt.scatter(nodes_cos, ref); plt.show()

    return retval


def iFSHT(coeffs, envmap_size, envmap_format='latlong', reduction_type='right'):
    if reduction_type != 'right':
        raise NotImplemented()

    degrees = int(np.sqrt(8*coeffs.shape[0])/2. - 1)

    ch = coeffs.shape[1]
    envmap = EnvironmentMap(np.zeros((envmap_size, envmap_size*2, ch)), envmap_format)
    envmap.data = envmap.data.astype(np.complex128)

    P, _ = _getP(envmap, degrees)

    i = 0
    for l in tqdm(range(degrees + 1)):
        for m in range(0, l + 1):
            for c in range(ch):
                envmap.data[:,m,c] += P[:,i]*coeffs[i,c]
            i += 1

    #import pdb; pdb.set_trace()

    envmap.data = np.fft.ifft(envmap.data, axis=1).real

    return envmap


def _getP(envmap, degrees):
    from cffi import FFI
    ffi = FFI()
    ffi.cdef("""
        void generateAssociatedLegendreFactors(const float N, float *data_out, const float * nodes, const unsigned int num_nodes);
    """)
    C = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)), "spharm_tools.dll"))

    x, y, z, valid = envmap.worldCoordinates()
    theta = np.arctan2(x, -z)
    phi = np.arccos(y).ravel().copy()

    nodes = phi.reshape(envmap.data.shape[:2])[:,0]
    P = np.empty((nodes.size, int((2*(degrees + 1)+1)**2/8)), dtype=np.float32) # square matrix?
    nodes_cos = np.cos(nodes).astype(np.float32).copy()

    import time; ts = time.time()
    print("Generating {}x{}={} vals".format(P.shape[0], P.shape[1], P.size))

    data_ptr = ffi.cast("float *", P.ctypes.data)
    nodes_cos_ptr = ffi.cast("float *", nodes_cos.ctypes.data)
    C.generateAssociatedLegendreFactors(degrees + 1, data_ptr, nodes_cos_ptr, nodes.size)
    P = P.reshape([envmap.data.shape[0], int((2*(degrees + 1)+1)**2/8)])
    print("Done in {:.3f}s".format(time.time() - ts))

    return P, nodes


def _getRefP(nodes_cos, degrees):
    #ts = time.time()
    from scipy.special import lpmv
    k = 0
    ref = np.zeros((int((2*(degrees + 1)+1)**2/8) * nodes_cos.size, ), dtype=np.float32)
    for no in nodes_cos:
        for l in range(degrees + 1):
            for m in range(l + 1):
                ref[k] = lpmv(m, l, no)
                k += 1
    #print("Temps manuel: {}s".format(time.time() - ts))
    ref = ref.reshape([nodes_cos.size, int((2*(degrees + 1)+1)**2/8)])

    return ref


def sphericalHarmonicTransform(envmap, degrees=None, reduction_type='right'):
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


def inverseSphericalHarmonicTransform(coeffs, envmap_height=512, format_='latlong', reduction_type='right'):
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
        raise NotImplemented()
    elif reduction_type == 'right':
        return coeffs[_triangleRightSide(degrees)]

    raise Exception('unknown reduction_type')


def addRedundantCoeffs(coeffs, reduction_type):

    degrees = round(np.asscalar(1./2. * (np.sqrt(8*coeffs.shape[0]+1) - 1))) - 1
    retval = np.empty(((degrees + 1)**2, *coeffs.shape[1:]), dtype=np.complex128)

    if reduction_type is None:
        return coeffs
    elif reduction_type == 'image_real':
        raise NotImplemented()
    elif reduction_type == 'right':
        retval[_triangleRightSide(degrees)] = coeffs
        for i in set(range((degrees + 1)**2)) - set(_triangleRightSide(degrees)):
            l = np.sqrt(i).astype('int')
            m = abs((l) - (i - l**2))
            retval[i] = (-1)**m * np.conj(retval[l**2 + l + m])
        return retval

    raise Exception('unknown reduction_type')


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    e = EnvironmentMap('envmap.exr', 'angular')
    e.resize((256, 256))
    e.convertTo('latlong')

    # P, nodes = _getP(e, 15)
    # refP = _getRefP(np.cos(nodes), 15)

    # for i in range(P.shape[1] - 5, P.shape[1]):
    #     plt.plot(np.linspace(-1, 1, P.shape[0]), P[:,i], label="{}".format(i))
    #     plt.plot(np.linspace(-1, 1, P.shape[0]), refP[:,i], label="ref{}".format(i))
    # plt.legend();
    # plt.show()

    # import pdb; pdb.set_trace()

    coeffs_fsht = FSHT(e.copy(), 25)

    coeffs = sphericalHarmonicTransform(e, 25)

    err_f = []
    err = []
    for degrees in range(30):
        db_coef = int((2*(degrees + 1)+1)**2/8)
        er_fsht = iFSHT(coeffs_fsht[:db_coef,:], 256)
        err_f.append(np.sum((er_fsht.data*0.029 - e.data)**2))
        er = inverseSphericalHarmonicTransform(coeffs[:db_coef,:], 256)
        err.append(np.sum((er.data - e.data)**2))
    plt.subplot(2,2,1); plt.imshow(e.data)
    plt.subplot(2,2,2); plt.imshow(er.data)
    plt.subplot(2,2,3); plt.imshow(er_fsht.data * 0.029)
    plt.subplot(2,2,4); plt.plot(range(30), err, label="Normal"); plt.plot(range(30), err_f, label="Fast"); plt.legend();
    plt.show()
    import pdb; pdb.set_trace()