import pytest
import math
import numpy as np

from envmap import projections as t
from envmap import environmentmap as env


# pytest [-s] [-k test_projections_cube]

@pytest.mark.parametrize("coordinate_UV, coordinate_XYZR", 
    [
        # ( coordinate_UV(u,v), coordinate_XYZ(x,y,z,r) )
        
        # Diagonal
        ((0.1, 0.1), (float('nan'), float('nan'), float('nan'), False)),    
        ((0.25, 0.25), (-0.6666666666666666, 0.6666666666666666, -0.3333333333333333, True)),    
        ((0.5, 0.5), (0., -0.70710678, -0.70710678, True)),
        ((0.75, 0.75), (float('nan'), float('nan'), float('nan'), False)), 
        ((0.9, 0.9), (float('nan'), float('nan'), float('nan'), False)),

        # Random
        ((0.6, 0.3), (0.4574957109978137, 0.45749571099781405, -0.7624928516630234, True)),
        ((0.3, 0.6), (float('nan'), float('nan'), float('nan'), False)),
    ]
)
def test_projections_cube(coordinate_UV, coordinate_XYZR):
    X, Y, Z, R = coordinate_XYZR
    U, V = coordinate_UV

    # --- Singular float coordinates --- #
    if R == True:
        u, v = t.world2cube(X, Y, Z)
        assert type(u) == float
        assert type(v) == float
        assert u == pytest.approx(U, abs=1e-6)
        assert v == pytest.approx(V, abs=1e-6)

    x, y, z, r = t.cube2world(U, V)
    assert type(x) == float
    assert type(y) == float
    assert type(z) == float
    assert type(r) == bool
    assert r == R
    np.testing.assert_almost_equal(x, X, decimal=6)
    np.testing.assert_almost_equal(y, Y, decimal=6)
    np.testing.assert_almost_equal(z, Z, decimal=6)
        
    # --- Array of float coordinates --- #
    U_, V_ = np.array([U, U, U]), np.array([V, V, V])
    X_, Y_, Z_, R_ = np.array([X, X, X]), np.array([Y, Y, Y]), np.array([Z, Z, Z]), np.array([R, R, R])

    if R == True:    
        u_, v_ = t.world2cube(X_, Y_, Z_)
        assert type(u_) == np.ndarray
        assert type(v_) == np.ndarray
        assert u_ == pytest.approx(U_, abs=1e-6)
        assert v_ == pytest.approx(V_, abs=1e-6)

    x_, y_, z_, r_ = t.cube2world(U_, V_)
    assert type(x_) == np.ndarray
    assert type(y_) == np.ndarray
    assert type(z_) == np.ndarray
    assert type(r_) == np.ndarray
    assert (r_ == R_).all()
    np.testing.assert_almost_equal(x_, X_, decimal=6)
    np.testing.assert_almost_equal(y_, Y_, decimal=6)
    np.testing.assert_almost_equal(z_, Z_, decimal=6)


@pytest.mark.parametrize("format_", env.SUPPORTED_FORMATS)
def test_projections_pixel(format_):
    e = env.EnvironmentMap(64, format_, channels=2)

    # Meshgrid of Normalized Coordinates 
    u, v = e.imageCoordinates()

    # Meshgrid of M*N image Coordinates 
    cols = np.linspace(0, e.data.shape[1] - 1, e.data.shape[1])
    rows = np.linspace(0, e.data.shape[0] - 1, e.data.shape[0])
    U, V = np.meshgrid(cols, rows)

    x, y, z, valid = e.image2world(u, v)
    x_, y_, z_, valid_ = e.pixel2world(U, V)

    np.testing.assert_array_almost_equal(x[valid], x_[valid_], decimal=5)
    np.testing.assert_array_almost_equal(y[valid], y_[valid_], decimal=5)
    np.testing.assert_array_almost_equal(z[valid], z_[valid_], decimal=5)
    np.testing.assert_array_equal(valid, valid_)

    # world2pixel(x,y,z)
    U_, V_ = e.world2pixel(x, y, z)

    np.testing.assert_array_equal(U_[valid_], U[valid_])
    np.testing.assert_array_equal(V_[valid_], V[valid_])


@pytest.mark.parametrize("format_", env.SUPPORTED_FORMATS)
def test_projections_image(format_):

    e = env.EnvironmentMap(64, format_, channels=2)

    # Meshgrid of Normalized Coordinates 
    cols = np.linspace(0, 1, e.data.shape[1]*2 + 1)[1::2]
    rows = np.linspace(0, 1, e.data.shape[0]*2 + 1)[1::2]
    u, v = np.meshgrid(cols, rows)

    # image2world(U,V)
    x, y, z, valid = e.image2world(u, v)

    # world2image(x,y,z)
    u_, v_ = e.world2image(x, y, z)

    np.testing.assert_array_almost_equal(u[valid], u_[valid], decimal=5)
    np.testing.assert_array_almost_equal(v[valid], v_[valid], decimal=5)
