import numpy as np


def rotx(theta):
    """
    Produces a counter-clockwise 3D rotation matrix around axis X with angle `theta` in radians.
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]], dtype='float64')


def roty(theta):
    """
    Produces a counter-clockwise 3D rotation matrix around axis Y with angle `theta` in radians.
    """
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]], dtype='float64')


def rotz(theta):
    """
    Produces a counter-clockwise 3D rotation matrix around axis Z with angle `theta` in radians.
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]], dtype='float64')


def rot(theta=(0,0,0)):
    """
    Produces a counter-clockwise 3D rotation matrix around axis X, Y and Z with angles `theta` in radians.
    """
    return np.dot(rotz(theta[2]), np.dot(roty(theta[1]), rotx(theta[0])))
