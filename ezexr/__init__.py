import os
import sys
import array
import time
import warnings

import numpy as np

cffi_def = """
float* readEXRfloat(const char filename[], char ***channel_names, int *width, int *height, int *nb_channels);
float* writeEXRfloat(const char filename[], const char *channel_names[], const float *data, int width, int height, int nb_channels);
"""

if os.name == 'nt':
    try:
        from cffi import FFI
        ffi = FFI()
        ffi.cdef(cffi_def)
        to_precache = ["libstdc++-6.dll", "libgcc_s_sjlj-1.dll", "libzlib.dll", "libHalf.dll", "libIex-2_2.dll",
                    "libIlmThread-2_2.dll", "libImath-2_2.dll", "libIlmImf-2_2.dll"]
        [ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)), x)) for x in to_precache]
        C = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)), "wrapper.dll"))
    except Exception as e:
        print("exr functionalities will not work, could not load dll: {}".format(e))
else:
    try:
        import OpenEXR
        import Imath
    except ImportError:
        if sys.platform == "darwin":
            from cffi import FFI
            ffi = FFI()
            ffi.cdef(cffi_def)
            C = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)), "wrapper.dylib"))
        else:
            raise


def imread_raw_custom_(filename):
    width = ffi.new("int*")
    height = ffi.new("int*")
    nb_channels = ffi.new("int*")

    cn = ffi.new("char***", ffi.new("char**", ffi.NULL))
    fn = ffi.new("char[]", bytes(filename, 'ascii'))
    ret = C.readEXRfloat(fn, cn, width, height, nb_channels)

    width = width[0]
    height = height[0]
    nb_channels = nb_channels[0]

    vals = np.frombuffer(ffi.buffer(ret, width*height*nb_channels*4), dtype=np.float32).reshape([height, width, nb_channels])
    channels = [ffi.string(cn[0][i]).decode('ascii') for i in range(nb_channels)]

    return vals, channels


def imread(filename, bufferImage=None, rgb=True):
    """
    Read an .exr image and returns a numpy matrix or a dict of channels.

    Does not support .exr with varying channels sizes.

    :bufferImage: If not None, then it should be a numpy array
                  of a sufficient size to contain the data.
                  If it is None, a new array is created and returned.
    :rgb: If True: tries to get the RGB(A) channels as an image
          If False: Returns all channels independently
          If "hybrid": "<identifier>.[R|G|B|A|X|Y|Z]" -> merged to an image
    """
    # Check if we should use the custom wrapper
    if 'OpenEXR' not in globals():
        if rgb is not True:
            raise NotImplemented()

        if bufferImage:
            warnings.warn("Buffer passing not supported yet with custom wrapper", RuntimeWarning)
        im, ch = imread_raw_custom_(filename)
        if len(ch) == 1:
            return im

        channelsToUse = ('R', 'G', 'B', 'A') if 'A' in ch else ('R', 'G', 'B')
        ordering = [ch.index(x) for x in channelsToUse]

        return im[:,:,ordering]

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

    if rgb is True:
        # Create the read buffer if needed
        data = bufferImage if bufferImage is not None else np.empty((h, w, nc), dtype=pixformat_mapping[dtGlobal.v])

        if nc == 1:  # Greyscale
            cname = list(header['channels'].keys())[0]
            data = np.fromstring(f.channel(cname), dtype=pixformat_mapping[dtGlobal.v]).reshape(h, w, 1)
        else:
            assert 'R' in header['channels'] and 'G' in header['channels'] and 'B' in header['channels'], "Not a grayscale image, but no RGB data!"
            channelsToUse = ('R', 'G', 'B', 'A') if 'A' in header['channels'] else ('R', 'G', 'B')
            nc = len(channelsToUse)
            for i, c in enumerate(channelsToUse):
                # Check the data type
                dt = header['channels'][c].type
                if dt.v != dtGlobal.v:
                    data[:, :, i] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w)).astype(pixformat_mapping[dtGlobal.v])
                else:
                    data[:, :, i] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w))
    else:
        data = {}

        for i, c in enumerate(header['channels']):
            dt = header['channels'][c].type
            data[c] = np.fromstring(f.channel(c), dtype=pixformat_mapping[dt.v]).reshape((h, w))

        if rgb == "hybrid":
            ordering = {key: i for i, key in enumerate("RGBAXYZ")}

            new_data = {}
            for c in data.keys():

                ident = c.split(".")[0]
                try:
                    chan = c.split(".")[1]
                except IndexError:
                    chan = "R"

                if ident not in new_data:
                    all_chans = [x.split(".")[1] for x in data if x.startswith(ident + ".")]
                    nc = len(all_chans)
                    new_data[ident] = np.empty((h, w, nc), dtype=np.float32)
                    for i, chan in enumerate(sorted(all_chans, key=lambda v: ordering.get(v, len(ordering)))):
                        new_data[ident][:,:,i] = data["{}.{}".format(ident, chan)].astype(new_data[ident].dtype)

            data = new_data

    f.close()
    
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
    if arr.ndim == 3:
        h, w, d = arr.shape
    elif arr.ndim == 2:
        h, w = arr.shape
        d = 1
    else:
        raise Exception("Could not understand dimensions in array.")

    # Check if we should use the custom wrapper
    if 'OpenEXR' not in globals():
        if d == 1:
            cl = "Y"
        elif d == 3:
            cl = "RGB"
        elif d == 4:
            cl = "RGBA"
        cl = [ffi.new("char*", bytes(x, "ascii")) for x in cl]

        #import pdb; pdb.set_trace()
        fn = ffi.new("char[]", bytes(filename, 'ascii'))
        cn = ffi.new("char*[]", cl)
        data_np = np.ascontiguousarray(arr.transpose([2,0,1]).astype("float32"))
        data = ffi.cast("float*", data_np.ctypes.data)
        print(fn, cn, data, w, h, d)
        C.writeEXRfloat(fn, cn, data, w, h, d)

        return

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

    out.close()


imsave = imwrite

__all__ = ['imread', 'imwrite', 'imsave']
