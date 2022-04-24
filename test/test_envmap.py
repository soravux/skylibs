import numpy as np
from itertools import product
import pytest
from skimage.transform import resize
from scipy.ndimage import binary_erosion
from envmap import EnvironmentMap, rotation_matrix
from envmap.environmentmap import SUPPORTED_FORMATS


np.random.seed(31415926)


# pytest [-s] [-k test_convert]


def get_envmap(sz, up_factor, format_, channels=3):
    e = EnvironmentMap(sz, format_, channels=channels)
    e.data = np.random.rand(e.data.shape[0], e.data.shape[1], channels)
    if up_factor != 1.:
        e.data = resize(e.data, [up_factor*x for x in e.data.shape[:2]])
    e.setBackgroundColor(0.)
    return e


@pytest.mark.parametrize("envmap_type,in_sz,out_sz", product(SUPPORTED_FORMATS, [512, 431, 271], [512, 431, 271]))
def test_resize_integer(envmap_type, in_sz, out_sz):
    e = get_envmap(in_sz, 1, envmap_type, 1)
    old_energy = e.data.mean()
    e = e.copy().resize(out_sz, debug=True)
    new_energy = e.data.mean()
    print("Energy difference: {:.04f}".format(np.abs(new_energy/old_energy - 1.)))
    assert np.abs(new_energy/old_energy - 1.) < 1e-3


@pytest.mark.parametrize("src_format,tgt_format", product(SUPPORTED_FORMATS, SUPPORTED_FORMATS))
def test_convert(src_format, tgt_format):
    e_src = get_envmap(16, 6, src_format)

    # remove everything not in the sky if src or tgt format is sky-only
    if src_format[:3] == "sky" or tgt_format[:3] == "sky":
        _, y, _, _ = e_src.worldCoordinates()
        e_src.data[np.tile(y[:,:,None], (1, 1, 3)) < 0] = 0.

    sa_src = e_src.solidAngles()
    old_energy = np.nansum(sa_src[:,:,None]*e_src.data)

    e_tgt = e_src.copy().convertTo(tgt_format)
    sa_tgt = e_tgt.solidAngles()
    new_energy = np.nansum(sa_tgt[:,:,None]*e_tgt.data)

    assert new_energy/old_energy - 1. < 1
    print("Energy difference in convertTo: {:.08f}".format(new_energy/old_energy - 1.))

    recovered = e_tgt.copy().convertTo(src_format)
    recovered_energy = np.nansum(sa_src[:,:,None]*recovered.data)

    assert recovered_energy/old_energy - 1. < 1
    print("Recovered energy difference: {:.08f}".format(recovered_energy/old_energy - 1.))


@pytest.mark.parametrize("format_", SUPPORTED_FORMATS)
def test_convert_self(format_):
    e_src = get_envmap(16, 6, format_)

    e_tgt = e_src.copy().convertTo(format_)
    #from matplotlib import pyplot as plt
    #plt.imshow(e_tgt.data[:,:,0] - e_src.data[:,:,0]); plt.colorbar(); plt.title(format_); plt.show()
    assert np.nanmax(np.abs(e_tgt.data - e_src.data)) < 1e-4


@pytest.mark.parametrize("format_", SUPPORTED_FORMATS)
def test_project_embed(format_):
    e = get_envmap(16, 6, format_)

    dcm = rotation_matrix(azimuth=0./180*np.pi,
                        elevation=-45./180*np.pi,
                        roll=0./180*np.pi)
    crop = e.project(vfov=85., # degrees
                    rotation_matrix=dcm,
                    ar=4/3,
                    resolution=(640, 480),
                    projection="perspective",
                    mode="normal")

    mask = e.project(vfov=85., # degrees
                    rotation_matrix=dcm,
                    ar=4/3,
                    resolution=(640, 480),
                    projection="perspective",
                    mode="mask") > 0.9

    e_embed = EnvironmentMap(e.data.shape[0], format_, channels=1)
    e_embed = e_embed.embed(vfov=85.,
                            rotation_matrix=dcm,
                            image=crop)

    e_embed.data[~np.isfinite(e_embed.data)] = 0.
    recovered = mask[:,:,None]*e_embed.data
    source = mask[:,:,None]*e.data
    # from matplotlib import pyplot as plt
    # plt.subplot(141); plt.imshow(crop) # mask.astype('float32'))
    # plt.subplot(142); plt.imshow(recovered)
    # plt.subplot(143); plt.imshow(source); plt.title(format_)
    # plt.subplot(144); plt.imshow(np.abs(recovered - source)); plt.colorbar()
    # plt.show()
    
    assert np.mean(np.abs(recovered - source)) < 1e-1

    # edges are not pixel-perfect, remove boundary for check
    mask = binary_erosion(mask)
    recovered = mask[:,:,None]*e_embed.data
    source = mask[:,:,None]*e.data
    # from matplotlib import pyplot as plt
    # plt.subplot(131); plt.imshow(recovered)
    # plt.subplot(132); plt.imshow(source); plt.title(format_)
    # plt.subplot(133); plt.imshow(np.abs(recovered - source)); plt.colorbar()
    # plt.show()
    assert np.max(np.abs(recovered - source)) < 0.15


@pytest.mark.parametrize("format_,mode,colorspace", product(SUPPORTED_FORMATS, ["ITU BT.601", "ITU BT.709", "mean"], ["sRGB", "linear"]))
def test_intensity(format_, mode, colorspace):
    e = get_envmap(16, 6, format_, channels=3)
    e.toIntensity(mode=mode, colorspace=colorspace)

    assert e.data.shape[2] == 1
