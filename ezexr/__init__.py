import sys
import array

import OpenEXR
import Imath

import numpy as np


def imread(filename):
    """
    Read an .exr image and returns a numpy matrix.
    TODO:
      - Support Alpha channel (and others)
      - Support Greyscale
    """
    # Open the input file
    f = OpenEXR.InputFile(filename)

    # Compute the size
    dw = f.header()['dataWindow']
    h, w = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = [ array.array('f', f.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
    return np.dstack((R, G, B)).reshape(h, w, 3)


def imwrite(filename, arr):
    """
    Write an .exr file from an input <arr>ay.
    TODO
    """
    h, w, d = arr.shape
    # Convert to strings
    (R, G, B) = [ array.array('f', Chan).tostring() for Chan in (arr[:,:,1], arr[:,:,2], arr[:,:,3]) ]

    # Write the three color channels to the output file
    out = OpenEXR.OutputFile(filename, OpenEXR.Header(h, w))
    out.writePixels({'R' : R, 'G' : G, 'B' : G })


imsave = imwrite

__all__ = ['imread', 'imwrite', 'imsave']
