import sys
import array
import warnings

import OpenEXR
import Imath
import time

import numpy as np


def imread(filename, bufferImage=None):
    """
    Read an .exr image and returns a numpy matrix.
    If bufferImage is not None, then it should be a numpy array
    of a sufficient size to contain the data.
    If it is None, a new array is created and returned.

.. todo::

    * Support Alpha channel (and others)

    """
    # Open the input file
    f = OpenEXR.InputFile(filename)

    # Get the header (we store it in a variable because this function read the file each time it is called)
    header = f.header()

    # Compute the size
    dw = header['dataWindow']
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1

    # Use the attribute "v" of PixelType objects because they have no __eq__
    pixformat_mapping = {Imath.PixelType(Imath.PixelType.FLOAT).v: np.float32,
                            Imath.PixelType(Imath.PixelType.HALF).v: np.float16,
                            Imath.PixelType(Imath.PixelType.UINT).v: np.uint32}
    
    # Get the number of channels
    nc = len(header['channels'])
    # Check the data type
    dtGlobal = list(header['channels'].values())[0].type
    
    # Create the read buffer if needed
    data = bufferImage if bufferImage is not None else np.empty((h, w, nc), dtype=pixformat_mapping[dtGlobal.v])
        
    if nc == 1:  # Greyscale
        cname = list(header['channels'].keys())[0]
        data = np.fromstring(f.channel(cname), dtype=pixformat_mapping[dtGlobal.v]).reshape(h, w, 1)
    else:
        assert 'R' in header['channels'] and 'G' in header['channels'] and 'B' in header['channels'], "Not a grayscale image, but no RGB data!"
        channelsToUse = ('R', 'G', 'B', 'A') if 'A' in header['channels'] else ('R', 'G', 'B')
        nc = len(channelsToUse)
        for i,c in enumerate(channelsToUse):
            # Check the data type
            dt = header['channels'][c].type
            data[:, :, i] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w))
            if dt.v != dtGlobal.v:
                data[:, :, i] = data[:, :, i].astype(pixformat_mapping[dtGlobal.v])
    
    return data


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
        # Default : Auto detect
        arr_fin = arr[np.isfinite(arr)]
        the_max = np.abs(arr_fin).max()
        the_min = np.abs(arr_fin[arr_fin > 0]).min()

        if the_max <= 65504. and the_min >= 1e-7:
            pixformat = 'HALF'
        elif the_max < 3.402823e+38 and the_min >= 1.18e-38:
            pixformat = 'FLOAT'
        else:
            raise Exception('Could not convert array into exr without loss of information '
                            '(a value would be rounded to infinity or 0)')
        warnings.warn("imwrite received an array with dtype={}, which cannot be saved in EXR format."
                      "Will fallback to {}, which can represent all the values in the array.".format(arr.dtype, pixformat), RuntimeWarning)

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
