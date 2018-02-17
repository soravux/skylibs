import os
import subprocess

import numpy as np
from scipy import misc as scipy_io


__version__ = "0.2"


try:
    import ezexr
except ImportError as e:
    print("Could not import exr module:", e)

try:
    import imageio
except ImportError as e:
    print("Could not import hdr module:", e)


def imwrite(data, filename):
    _, ext = os.path.splitext(filename.lower())
    if ext == '.exr':
        ezexr.imwrite(filename, data)
    elif ext in ['.hdr', '.pic']:
        _hdr_write(filename, data)
    else:
        scipy_io.imsave(filename, data)


def imsave(filename, data):
    imwrite(data, filename)


def imread(filename, format_="float32"):
    """Reads an image. Supports exr, hdr, cr2, tiff, jpg, png and
    everything SciPy/PIL supports.

    :filename: file path.
    :format_: format in which to return the value. If set to "native", the
              native format of the file will be given (e.g. uint8 for jpg).
    """
    ldr = False
    _, ext = os.path.splitext(filename.lower())

    if ext == '.exr':
        im = ezexr.imread(filename)
    elif ext in ['.hdr', '.pic']:
        im = _hdr_read(filename)
    elif ext in ['.cr2', '.nef', '.raw']:
        im = _raw_read(filename)
    elif ext in ['.tiff', '.tif']:
        try:
            import tifffile as tiff
        except ImportError:
            print('Install tifffile for better tiff support. Fallbacking to '
                  'scipy.')
            im = scipy_io.imread(filename)
        else:
            im = tiff.imread(filename)
    else:
        im = scipy_io.imread(filename)
        ldr = True

    if format_ == "native":
        return im
    elif ldr and not 'int' in format_:
        return im.astype(format_) / 255.
    else:
        return im.astype(format_)


def _raw_read(filename):
    """Calls the dcraw program to unmosaic the raw image."""
    fn, _ = os.path.splitext(filename.lower())
    target_file = "{}.tiff".format(fn)
    if not os.path.exists(target_file):
        ret = subprocess.call('dcraw -v -T -4 -t 0 -j {}'.format(filename))
        if ret != 0:
            raise Exception('Could not execute dcraw. Make sure the executable'
                            ' is available.')
    try:
        import tifffile as tiff
    except ImportError:
        raise Exception('Install tifffile to read the converted tiff file.')
    else:
        return tiff.imread(target_file)


def _hdr_write(filename, data, **kwargs):
    """Write a Radiance hdr file.
Refer to the ImageIO API ( http://imageio.readthedocs.io/en/latest/userapi.html
) for parameter description."""

    imageio.imwrite(filename, data, **kwargs)


def _hdr_read(filename, **kwargs):
    """Read a Radiance hdr file.
Refer to the ImageIO API ( http://imageio.readthedocs.io/en/latest/userapi.html
) for parameter description."""
    return imageio.imread(filename, **kwargs)


__all__ = ['imwrite', 'imsave', 'imread']
