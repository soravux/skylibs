import pytest
import numpy as np

from envmap import environmentmap as env

def test_projections_pixel():
    
    env_formats = [ 'angular', 'skyangular', 'latlong', 'skylatlong', 'sphere', 'cube' ]

    for format in env_formats:
        e = env.EnvironmentMap(8, format, channels=2)

        # Meshgrid of Normalized Coordinates 
        cols = np.linspace(0, 1, e.data.shape[1]*2 + 1)
        rows = np.linspace(0, 1, e.data.shape[0]*2 + 1)
        cols = cols[1::2]
        rows = rows[1::2]
        u, v = np.meshgrid(cols, rows)

        # Meshgrid of M*N image Coordinates 
        cols = np.linspace(0, e.data.shape[1]-1, e.data.shape[1])
        rows = np.linspace(0, e.data.shape[0]-1, e.data.shape[0])
        U, V = np.meshgrid(cols, rows)

        # pixel2world(U,V)
        x, y, z, v = e.image2world(u,v) 
        x_, y_, z_, v_ = e.pixel2world(U, V)

        np.testing.assert_array_almost_equal(x, x_, decimal=6)
        np.testing.assert_array_almost_equal(y, y_, decimal=6)
        np.testing.assert_array_almost_equal(z, z_, decimal=6)
        np.testing.assert_array_equal(v,v_)

        # world2pixel(x,y,z)
        U_, V_ = e.world2pixel(x,y,z)

        np.testing.assert_array_equal(U_, U)
        np.testing.assert_array_equal(V_, V)
