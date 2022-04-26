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
    old = e.data.copy()
    e = e.copy().resize(out_sz, debug=True)
    new_energy = e.data.mean()
    print("Energy difference: {:g}".format(np.abs(new_energy/old_energy - 1.)))
    assert np.abs(new_energy/old_energy - 1.) < 5e-3


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

@pytest.mark.parametrize("format_,normal,channels", product(SUPPORTED_FORMATS, [[0, 1, 0], [1, 0, 0], [0, 0, -1], [0.707, 0.707, 0], "rand"], [1, 3, 5, -3]))
def test_set_hemisphere(format_, normal, channels):
    if channels < 0:
        value = np.asarray(np.random.rand())
        channels = np.abs(channels)
    else:
        value = np.random.rand(channels)

    if normal == "rand":
        normal = np.random.rand(3) + 1e-4
        normal /= np.linalg.norm(normal)
    else:
        normal = np.asarray(normal, dtype=np.float32)

    e = EnvironmentMap(128, format_, channels=channels)
    e.setHemisphereValue(normal, value)

    if value.size != e.data.shape[2]:
        value = np.tile(value, (e.data.shape[2],))

    u, v = e.world2image(normal[0:1], normal[1:2], normal[2:3])
    h, w = np.floor(v*e.data.shape[0]).astype('int16'), np.floor(u*e.data.shape[1]).astype('int16')
    h, w = np.minimum(h, e.data.shape[0] - 1), np.minimum(w, e.data.shape[1] - 1)
    assert np.all(e.data[h, w, :].squeeze().tolist() == value.squeeze().tolist()) , "normal not set"

    # skip sky-* envmaps as they might not represent the opposite normal
    if "sky" in format_:
        return

    u, v = e.world2image(-normal[0:1], -normal[1:2], -normal[2:3])
    h, w = np.floor(v*e.data.shape[0]).astype('int16'), np.floor(u*e.data.shape[1]).astype('int16')
    h, w = np.minimum(h, e.data.shape[0] - 1), np.minimum(w, e.data.shape[1] - 1)

    # try:
    assert np.sum(np.abs(e.data[h, w, :])) == 0. , "opposite normal not zeros"
    # except:
    #     print(format_)
    #     from matplotlib import pyplot as plt
    #     plt.imshow(e.data)
    #     plt.show()
    #     import pdb; pdb.set_trace()


@pytest.mark.parametrize("format_,normal", product(SUPPORTED_FORMATS, [[0, 1, 0], [1, 0, 0], [0, 0, -1], [0.707, 0.707, 0], "rand"]))
def test_worldCoordinates_list(format_, normal):
    e = EnvironmentMap(128, format_)
    if normal == "rand":
        normal = np.random.rand(3) + 1e-4
        normal /= np.linalg.norm(normal)

    u, v = e.world2image(*normal)


@pytest.mark.parametrize("format_,normal", product(SUPPORTED_FORMATS, [[0, 1, 0],
                                                                       [1, 0, 0],
                                                                       [0, 0, -1],
                                                                       [0.707, 0.707, 0],
                                                                       [[0.707, 0],
                                                                        [0.707, 0],
                                                                        [0, -1]],
                                                                       "rand"]))
def test_worldCoordinates_ndarray(format_, normal):
    e = EnvironmentMap(128, format_)
    if normal == "rand":
        normal = np.random.rand(3) + 1e-4
        normal /= np.linalg.norm(normal)

    normal = np.asarray(normal)
    u, v = e.world2image(*normal)
