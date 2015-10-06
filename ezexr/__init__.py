import sys
import array
import warnings

import OpenEXR
import Imath

import numpy as np


def imread(filename):
    """
    Read an .exr image and returns a numpy matrix.

.. todo::

    * Support Alpha channel (and others)
    * Support Greyscale

    """
    # Open the input file
    f = OpenEXR.InputFile(filename)

    # Compute the size
    dw = f.header()['dataWindow']
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1

    # Use the attribute "v" of PixelType objects because they have no __eq__
    pixformat_mapping = {Imath.PixelType(Imath.PixelType.FLOAT).v: np.float32,
                            Imath.PixelType(Imath.PixelType.HALF).v: np.float16,
                            Imath.PixelType(Imath.PixelType.UINT).v: np.uint32}

    data = []
    nc = len(f.header()['channels'])
    if nc == 3:    # RGB
        for c in ('R', 'G', 'B'):
            # Check the data type
            dt = f.header()['channels'][c].type
            data.append(np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]))
    elif nc == 1:  # Greyscale
        cname = list(f.header()['channels'].keys())[0]
        # Check the data type
        dt = f.header()['channels'][cname].type
        data.append(np.fromstring(f.channel(cname), dtype=pixformat_mapping[dt.v]))

    return np.dstack(data).reshape(h, w, nc)


def imwrite(filename, arr, **params):
    """
    Write an .exr file from an input array.

    Optionnal params : 
    compression = 'NONE' | 'RLE' | 'ZIPS' | 'ZIP' | 'PIZ' | 'PXR24' (default PIZ)
    pixeltype = 'HALF' | 'FLOAT' | 'UINT' (default : dtype of the input array if float16, float32 or uint32, else float16)

.. todo::

      Test it

    """
    h, w, d = arr.shape

    compression = 'PIZ' if not 'compression' in params or\
                     params['compression'] not in ('NONE', 'RLE', 'ZIPS', 'ZIP', 'PIZ', 'PXR24') else params['compression']
    imath_compression = {'NONE' : Imath.Compression(Imath.Compression.NO_COMPRESSION),
                            'RLE' : Imath.Compression(Imath.Compression.RLE_COMPRESSION),
                            'ZIPS' : Imath.Compression(Imath.Compression.ZIPS_COMPRESSION),
                            'ZIP' : Imath.Compression(Imath.Compression.ZIP_COMPRESSION),
                            'PIZ' : Imath.Compression(Imath.Compression.PIZ_COMPRESSION),
                            'PXR24' : Imath.Compression(Imath.Compression.PXR24_COMPRESSION)}[compression]


    if 'pixeltype' in params and params['pixeltype'] in ('HALF', 'FLOAT', 'UINT'):
        # User-defined pixel type
        pixformat = params['pixeltype']
    elif arr.dtype == np.float32:
        pixformat = 'FLOAT'
    elif arr.dtype == np.uint32:
        pixformat = 'UINT'
    elif arr.dtype == np.float16:
        pixformat = 'HALF'
    else:
        # Default : half precision float
        pixformat = 'HALF'
        warnings.warn("imwrite received an array with dtype={}, which cannot be saved in EXR format. Will fallback to HALF-PRECISION.".format(arr.dtype), RuntimeWarning)

    imath_pixformat = {'HALF' : Imath.PixelType(Imath.PixelType.HALF),
                        'FLOAT' : Imath.PixelType(Imath.PixelType.FLOAT),
                        'UINT' : Imath.PixelType(Imath.PixelType.UINT)}[pixformat]
    numpy_pixformat = {'HALF' : 'float16',
                        'FLOAT' : 'float32',
                        'UINT' : 'uint32'}[pixformat]      # Not sure for the last one...

    # Convert to strings
    # TODO: Investigate the side-effects of the float cast
    R, G, B = [ x.astype(numpy_pixformat).tostring() for x in [arr[:,:,0], arr[:,:,1], arr[:,:,2]] ]
    #(R, G, B) = [ array.array('f', Chan).tostring() for Chan in (arr[:,:,0], arr[:,:,1], arr[:,:,2]) ]

    outHeader = OpenEXR.Header(w, h)
    outHeader['compression'] = imath_compression        # Apply compression
    for channel in outHeader['channels']:               # Apply pixel format
        outHeader['channels'][channel] = Imath.Channel(imath_pixformat, 1, 1)

    # Write the three color channels to the output file
    out = OpenEXR.OutputFile(filename, outHeader)
    out.writePixels({'R' : R, 'G' : G, 'B' : B })


imsave = imwrite

__all__ = ['imread', 'imwrite', 'imsave']
