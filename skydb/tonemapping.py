import numpy as np


def reinhard2002(self, envmap, scale=700):
    """Performs the Reinhard 2002 operator as described in
    Reinhard, Erik, et al. "Photographic tone reproduction for digital
    images." ACM Transactions on Graphics (TOG). Vol. 21. No. 3. ACM, 2002.

    :returns: 8-bits tone-mapped version of the environment map
    """
    data = envmap.data - envmap.data.min()
    return np.clip(scale * data / (1. + data), 0., 255.).astype('uint8')


def gamma(self, envmap, gamma=2.0, scale=255):
    """Performs a gamma compression: scale*V^(1/gamma) .

    :returns: 8-bits tone-mapped version of the environment map
    """
    data = envmap.data - envmap.data.min()
    return np.clip(scale * np.power(data, 1./gamma), 0., 255.).astype('uint8')