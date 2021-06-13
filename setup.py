import os
from setuptools import setup


if os.name == 'nt':
    extra_requires = ['cffi>=1.9.1']
    dependency_links = []
    package_data = {"ezexr": ["libHalf.dll",
                              "libIex-2_2.dll",
                              "libIlmImf-2_2.dll",
                              "libIlmThread-2_2.dll",
                              "libImath-2_2.dll",
                              "libzlib.dll",
                              "wrapper.dll",
                              "libgcc_s_sjlj-1.dll",
                              "libstdc++-6.dll"]}
else:
    extra_requires = ['openexr>=1.3.0']
    dependency_links = ['https://github.com/jamesbowman/openexrpython/tarball/master#egg=openexr-1.3.0']
    package_data = {}

setup(
    name='skylibs',
    description=('Tools to read, write, perform projections and handle LDR/HDR environment maps (IBL).'),
    author='Yannick Hold',
    author_email='yannickhold@gmail.com',
    license="LGPLv3",
    url='https://github.com/soravux/skylibs',
    version='0.6.2',
    packages=['ezexr', 'envmap', 'hdrio', 'hdrtools', 'hdrtools/tonemapping', 'skydb', 'tools3d'],
    package_data=package_data,
    include_package_data=True,
    install_requires=['imageio>=1.6', 'tqdm', 'pyshtools'].extend(extra_requires),
    dependency_links=dependency_links,
)

