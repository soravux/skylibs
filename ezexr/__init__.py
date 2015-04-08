import sys
import array

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

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R, G, B = [ np.fromstring(f.channel(x, FLOAT), dtype=np.float32) for x in ['R', 'G', 'B'] ]
    return np.dstack((R, G, B)).reshape(h, w, 3)


def imwrite(filename, arr, **params):
    """
    Write an .exr file from an input array.

    Optionnal params : 
    compression = 'NONE' | 'RLE' | 'ZIPS' | 'ZIP' | 'PIZ' | 'PXR24' (default PIZ)
    pixeltype = 'HALF' | 'FLOAT' | 'UINT' (default FLOAT)

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

    pixformat = 'FLOAT' if not 'pixeltype' in params or params['pixeltype'] not in ('HALF', 'FLOAT') else params['pixeltype']
    imath_pixformat = {'HALF' : Imath.PixelType(Imath.PixelType.HALF),
                        'FLOAT' : Imath.PixelType(Imath.PixelType.FLOAT),
                        'UINT' : Imath.PixelType(Imath.PixelType.UINT)}[pixformat]
    numpy_pixformat = {'HALF' : 'float16',
                        'FLOAT' : 'float32',
                        'UINT' : 'uint32'}[pixformat]      # Not sure for the last one...

    # Convert to strings
    # TODO: Investigate the side-effects of the float cast
    R, G, B = [ x.astype(numpy_pixformat).tobytes() for x in [arr[:,:,0], arr[:,:,1], arr[:,:,2]] ]
    #(R, G, B) = [ array.array('f', Chan).tostring() for Chan in (arr[:,:,0], arr[:,:,1], arr[:,:,2]) ]


    outHeader = OpenEXR.Header(h, w)
    outHeader['compression'] = imath_compression        # Apply compression
    for channel in outHeader['channels']:               # Apply pixel format
        outHeader['channels'][channel] = Imath.Channel(imath_pixformat, 1, 1)

    # Write the three color channels to the output file
    out = OpenEXR.OutputFile(filename, outHeader)
    out.writePixels({'R' : R, 'G' : G, 'B' : B })


imsave = imwrite

__all__ = ['imread', 'imwrite', 'imsave']
