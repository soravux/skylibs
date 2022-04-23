import numpy as np
import pytest
from envmap import EnvironmentMap
from envmap.environmentmap import SUPPORTED_FORMATS


# pytest [-s]


@pytest.mark.parametrize("envmap_type", SUPPORTED_FORMATS)
def test_resize_integer(envmap_type):
    e = EnvironmentMap(512, envmap_type)
    e.data += 1
    old_energy = e.data.mean()
    e = e.resize(256, debug=True)
    new_energy = e.data.mean()
    print(old_energy, new_energy)
    assert np.abs(new_energy/old_energy - 1.) < 1e-6


@pytest.mark.parametrize("envmap_type", SUPPORTED_FORMATS)
def test_resize_ratio(envmap_type):
    e = EnvironmentMap(431, envmap_type)
    e.data += 1
    old_energy = e.data.mean()
    e = e.resize(271, debug=True)
    new_energy = e.data.mean()
    assert np.abs(new_energy/old_energy - 1.) < 1e-6


@pytest.mark.parametrize("envmap_type", SUPPORTED_FORMATS)
def test_resize_upsample(envmap_type):
    e = EnvironmentMap(271, envmap_type)
    e.data += 1
    old_energy = e.data.mean()
    e = e.resize(431, debug=True)
    new_energy = e.data.mean()
    assert np.abs(new_energy/old_energy - 1.) < 1e-6
