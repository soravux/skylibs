import os

import numpy as np
from scipy import misc as scipy_io

try:
    import ezexr
except ImportError as e:
    print("Could not import exr module:", e)


def imwrite(data, filename):
    _, ext = os.path.splitext(filename.lower())
    if ext == '.exr':
        ezexr.imwrite(filename, data)
    elif ext == '.hdr':
        _hdr_write(filename, data)
    else:
        scipy_io.imsave(filename, data)


def imsave(filename, data):
    imwrite(data, filename)


def imread(filename):
    _, ext = os.path.splitext(filename.lower())
    if ext == '.exr':
        return ezexr.imread(filename)
    elif ext == '.hdr':
        return _hdr_read(filename)
    else:
        return scipy_io.imread(filename)


def _hdr_write(filename, data):
    """Reference: https://gist.github.com/edouardp/3089602 """
    # Assumes you have a np.array((height,width,3), dtype=float) as your HDR image
    data = data.astype('float')
     
    with open(filename, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with SkyLibs\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(bytes("-Y {0} +X {1}\n".format(data.shape[0], data.shape[1]), encoding="ascii"))

        brightest = np.maximum(np.maximum(data[...,0], data[...,1]), data[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 256.0 / brightest
        rgbe = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(data[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def _hdr_read(filename):
    """Read hdr file.

.. TODO:
   
    * Support axis other than -Y +X
"""
    with open(filename, "rb") as f:
        MAGIC = f.readline().strip()
        assert MAGIC == b'#?RADIANCE', "Wrong header found in {}".format(filename)

        comments = b""
        while comments[:6] != b"FORMAT":
            comments = f.readline().strip()
            assert comments[:3] != b"-Y ", "Could not find data format"
        assert comments == b'FORMAT=32-bit_rle_rgbe', "Format not supported"

        while comments[:3] != b"-Y ":
            comments = f.readline().strip()
        _, height, _, width = comments.decode("ascii").split(" ")
        height, width = int(height), int(width)

        rgbe = np.fromfile(f, dtype=np.uint8).reshape((height, width, 4))
        rgb = np.empty((height, width, 3), dtype=np.float)
        rgb[...,0] = np.ldexp(rgbe[...,0], rgbe[...,3].astype('int') - 128)
        rgb[...,1] = np.ldexp(rgbe[...,1], rgbe[...,3].astype('int') - 128)
        rgb[...,2] = np.ldexp(rgbe[...,2], rgbe[...,3].astype('int') - 128)
        # TODO: This will rescale all the values to be in [0, 1]. Find a way to retrieve the original values.
        rgb /= rgb.max()

    return rgb


__all__ = ['imwrite', 'imsave', 'imread']