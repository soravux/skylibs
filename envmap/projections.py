import numpy as np
from numpy import logical_and as land, logical_or as lor

from .rotations import *

def world2latlong(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a latitude-longitude map."""
    u = 1 + (1 / np.pi) * np.arctan2(x, -z)
    v = (1 / np.pi) * np.arccos(y)
    # because we want [0,1] interval
    u = u / 2
    return u, v


def world2skylatlong(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a sky-latitude-longitude map (the zenith hemisphere of a latlong map)."""
    u = 1 + (1 / np.pi) * np.arctan2(x, -z)
    v = (1 / np.pi) * np.arccos(y) * 2
    # because we want [0,1] interval
    u = u / 2
    return u, v


def world2angular(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    an angular map."""
    # world -> angular

    # take advantage of the division by zero handling of numpy
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    denum = (2 * np.pi * np.sqrt(x**2 + y**2))
    rAngular = np.arccos(-z) / denum
    v = np.atleast_1d(0.5 - rAngular * y)
    u = np.atleast_1d(0.5 + rAngular * x)
    
    u[~np.isfinite(rAngular)] = 0.5
    # handle [0, 0, -1]
    v[np.isnan(rAngular)] = 0.5
    # handle [0, 0, 1]
    v[np.isinf(rAngular)] = 0.

    if u.size == 1:
        return u.item(), v.item()

    return u, v


def latlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def skylatlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def angular2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for an angular map."""
    # angular -> world
    thetaAngular = np.arctan2(-2 * v + 1, 2 * u - 1)
    phiAngular = np.pi * np.sqrt((2 * u - 1)**2 + (2 * v - 1)**2)

    x = np.sin(phiAngular) * np.cos(thetaAngular)
    y = np.sin(phiAngular) * np.sin(thetaAngular)
    z = -np.cos(phiAngular)

    r = (u - 0.5)**2 + (v - 0.5)**2
    valid = r <= .25  # .5**2

    return x, y, z, valid


def skyangular2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a sky angular map."""
    # skyangular -> world
    thetaAngular = np.arctan2(-2 * v + 1, 2 * u - 1)  # azimuth
    phiAngular = np.pi / 2 * np.sqrt((2 * u - 1)**2 + (2 * v - 1)**2)  # zenith

    x = np.sin(phiAngular) * np.cos(thetaAngular)
    z = np.sin(phiAngular) * np.sin(thetaAngular)
    y = np.cos(phiAngular)

    r = (u - 0.5)**2 + (v - 0.5)**2
    valid = r <= .25  # .5^2

    return x, y, z, valid


def world2skyangular(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a sky angular map."""
    # world -> skyangular
    thetaAngular = np.arctan2(x, z)  # azimuth
    phiAngular = np.arctan2(np.sqrt(x**2 + z**2), y)  # zenith

    r = phiAngular / (np.pi / 2)

    u = 1. / 2 + r * np.sin(thetaAngular) / 2 
    v = 1. / 2 - r * np.cos(thetaAngular) / 2

    return u, v


def sphere2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for the sphere map."""
    u = u * 2 - 1
    v = v * 2 - 1

    # sphere -> world
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(u, -v)

    phi = np.zeros(theta.shape)
    valid = r <= 1
    phi[valid] = 2 * np.arcsin(r[valid])

    x = np.sin(phi) * np.sin(theta)
    y = np.sin(phi) * np.cos(theta)
    z = -np.cos(phi)
    return x, y, z, valid


def world2sphere(x, y, z):
    # world -> sphere

    # take advantage of the division by zero handling of numpy
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    
    denum = (2 * np.sqrt(x**2 + y**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.sin(.5 * np.arccos(-z)) / denum

    u = np.atleast_1d(.5 + r * x)
    v = np.atleast_1d(.5 - r * y)

    u[~np.isfinite(r)] = 0.5
    # handle [0, 0, -1]
    v[np.isnan(r)] = 0.5
    # handle [0, 0, 1]
    v[np.isinf(r)] = 0.

    if u.size == 1:
        return u.item(), v.item()

    return u, v


def world2cube(x, y, z):
    # world -> cube
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    z = np.atleast_1d(np.asarray(z))
    u = np.zeros(x.shape)
    v = np.zeros(x.shape)

    # forward
    indForward = np.nonzero(
        land(land(z <= 0, z <= -np.abs(x)), z <= -np.abs(y)))
    u[indForward] = 1.5 - 0.5 * x[indForward] / z[indForward]
    v[indForward] = 1.5 + 0.5 * y[indForward] / z[indForward]

    # backward
    indBackward = np.nonzero(
        land(land(z >= 0,  z >= np.abs(x)),  z >= np.abs(y)))
    u[indBackward] = 1.5 + 0.5 * x[indBackward] / z[indBackward]
    v[indBackward] = 3.5 + 0.5 * y[indBackward] / z[indBackward]

    # down
    indDown = np.nonzero(
        land(land(y <= 0,  y <= -np.abs(x)),  y <= -np.abs(z)))
    u[indDown] = 1.5 - 0.5 * x[indDown] / y[indDown]
    v[indDown] = 2.5 - 0.5 * z[indDown] / y[indDown]

    # up
    indUp = np.nonzero(land(land(y >= 0,  y >= np.abs(x)),  y >= np.abs(z)))
    u[indUp] = 1.5 + 0.5 * x[indUp] / y[indUp]
    v[indUp] = 0.5 - 0.5 * z[indUp] / y[indUp]

    # left
    indLeft = np.nonzero(
        land(land(x <= 0,  x <= -np.abs(y)),  x <= -np.abs(z)))
    u[indLeft] = 0.5 + 0.5 * z[indLeft] / x[indLeft]
    v[indLeft] = 1.5 + 0.5 * y[indLeft] / x[indLeft]

    # right
    indRight = np.nonzero(land(land(x >= 0,  x >= np.abs(y)),  x >= np.abs(z)))
    u[indRight] = 2.5 + 0.5 * z[indRight] / x[indRight]
    v[indRight] = 1.5 - 0.5 * y[indRight] / x[indRight]

    # bring back in the [0,1] intervals
    u = u / 3.
    v = v / 4.

    if u.size == 1:
        return u.item(), v.item()

    return u, v


def cube2world(u, v):
    u = np.atleast_1d(np.asarray(u))
    v = np.atleast_1d(np.asarray(v))
    
    # [u,v] = meshgrid(0:3/(3*dim-1):3, 0:4/(4*dim-1):4);
    # u and v are in the [0,1] interval, so put them back to [0,3]
    # and [0,4]
    u = u * 3
    v = v * 4

    x = np.zeros(u.shape)
    y = np.zeros(u.shape)
    z = np.zeros(u.shape)
    valid = np.zeros(u.shape, dtype='bool')

    # up
    indUp = land(land(u >= 1, u < 2), v < 1)
    x[indUp] = (u[indUp] - 1.5) * 2
    y[indUp] = 1
    z[indUp] = (v[indUp] - 0.5) * -2

    # left
    indLeft = land(land(u < 1, v >= 1), v < 2)
    x[indLeft] = -1
    y[indLeft] = (v[indLeft] - 1.5) * -2
    z[indLeft] = (u[indLeft] - 0.5) * -2

    # forward
    indForward = land(land(land(u >= 1, u < 2), v >= 1), v < 2)
    x[indForward] = (u[indForward] - 1.5) * 2
    y[indForward] = (v[indForward] - 1.5) * -2
    z[indForward] = -1

    # right
    indRight = land(land(u >= 2, v >= 1), v < 2)
    x[indRight] = 1
    y[indRight] = (v[indRight] - 1.5) * -2
    z[indRight] = (u[indRight] - 2.5) * 2

    # down
    indDown = land(land(land(u >= 1, u < 2), v >= 2), v < 3)
    x[indDown] = (u[indDown] - 1.5) * 2
    y[indDown] = -1
    z[indDown] = (v[indDown] - 2.5) * 2

    # backward
    indBackward = land(land(u >= 1, u < 2), v >= 3)
    x[indBackward] = (u[indBackward] - 1.5) * 2
    y[indBackward] = (v[indBackward] - 3.5) * 2
    z[indBackward] = 1

    # normalize
    # np.hypot(x, y, z) #sqrt(x.^2 + y.^2 + z.^2);
    norm = np.sqrt(x**2 + y**2 + z**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = x / norm
        y = y / norm
        z = z / norm

    # return valid indices
    valid_ind = lor(
        lor(lor(indUp, indLeft), lor(indForward, indRight)), lor(indDown, indBackward))
    valid[valid_ind] = 1

    if x.size == 1:
        return x.item(), y.item(), z.item(), valid.item()

    return x, y, z, valid


def ocam2world(u, v, ocam_calibration):
    """ Project a point (u, v) in omnidirectional camera space to 
    (x, y, z) point in world coordinate space."""

    # Step 0. De-Normalize coordinates to interval defined by the MxN image
    # Where u=cols(x),  v=rows(y)
    width_cols = ocam_calibration['width']
    height_rows = ocam_calibration['height']

    v = np.floor(v*height_rows).astype(int)
    u = np.floor(u*width_cols).astype(int)

    # Step 1. Center & Skew correction
    # M = affine matrix [c,d,xc; e,1,yc; 0,0,1]
    # p' = M^-1 * (p)
    M = ocam_calibration['affine_3x3']
    F = ocam_calibration['F']

    # Inverse: M_ = M^-1
    M_ = np.linalg.inv(M)

    # Affine transform
    w = np.ones_like(u)
    assert u.shape == v.shape
    save_original_shape = u.shape
    p_uvw = np.array((v.reshape(-1),u.reshape(-1),w.reshape(-1)))
    p_xyz = np.matmul(M_,p_uvw)

    # Add epsilon to mitigate NAN
    p_xyz[ p_xyz==0 ] = np.finfo(p_xyz.dtype).eps

    # Step 2. Get unit-sphere world coordinate z
    # Distance to center of image: p = sqrt(X^2 + Y^2)
    p_z = np.linalg.norm(p_xyz[0:2], axis=0)
    # Convert to z-coordinate with p_z = F(p)
    p_xyz[2] = F(p_z)

    # Step 3. Normalize x,y,z to unit length of 1 (unit sphere)
    p_xyz = p_xyz / np.linalg.norm(p_xyz, axis=0)

    # Step 4. Fix coordinate system alignment 
    # (rotate -90 degrees around z-axis)
    # (x,y,z) -> (x,-z,y) as +y is up (not +z)
    p_xyz = np.matmul(rotz(np.deg2rad(-90)),p_xyz)
    x,y,z = (
        p_xyz[0].reshape(save_original_shape),
        -p_xyz[2].reshape(save_original_shape),
        -p_xyz[1].reshape(save_original_shape)
    )

    valid = np.ones(x.shape, dtype='bool')
    return x,y,z, valid


def world2ocam(x, y, z, ocam_calibration):
    """ Project a point (x, y, z) in world coordinate space to 
    a (u, v) point in omnidirectional camera space."""

    # Step 1. Center & Skew correction
    # M = affine matrix [c,d; e,1]
    # T = translation vector
    # p' = M^-1 * (p - T)
    M = ocam_calibration['affine_3x3']
    F = ocam_calibration['F']

    assert x.shape == y.shape and x.shape == z.shape, f'{x.shape} == {y.shape} == {z.shape}'
    save_original_shape = x.shape

    # Step 2. Fix coordinate system alignment
    # (x,y,z) -> (x,z,-y) as +z is up (not +y)
    p_xyz = np.array((x.reshape(-1),y.reshape(-1),-z.reshape(-1)))
                      
    # Add epsilon to mitigate NAN
    p_xyz[ p_xyz==0 ] = np.finfo(p_xyz.dtype).eps
                      
    # (rotate 90 degrees around z-axis)
    p_xyz = np.array((p_xyz[0], p_xyz[2], -p_xyz[1]))
    p_xyz = np.matmul(rotz(np.deg2rad(90)),p_xyz)

    # Step 3. 3D to 2D
    m = p_xyz[2] / np.linalg.norm(p_xyz[0:2], axis=0)

    def poly_inverse(y):
        F_ = F.copy()
        F_.coef[1] -=y
        F_r = F_.roots()
        F_r = F_r[ (F_r >= 0) & (F_r.imag == 0) ]
        if len(F_r)>0:
            return F_r[0].real
        else:
            return np.nan
    m_ = np.vectorize(poly_inverse)(m)

    uvw = p_xyz / np.linalg.norm(p_xyz[0:2], axis=0) * m_

    # Step 4. Affine transform for center and skew
    uvw[2,:] = 1
    uvw = np.nan_to_num(uvw)
    uvw = np.matmul(M, uvw)
    u,v = uvw[1].reshape(save_original_shape), uvw[0].reshape(save_original_shape)

    # Step 3. Normalize coordinates to interval [0,1]
    # Where u=cols(x),  v=rows(y)
    width_cols = ocam_calibration['width']
    height_rows = ocam_calibration['height']
    u = (u+0.5) / width_cols
    v = (v+0.5) / height_rows

    return u,v
