from setuptools import setup

setup(
    name='skylibs',
    version='0.1',
    packages=['ezexr', 'envmap', 'hdrio', 'hdrtools', 'hdrtools/tonemapping', 'skydb', 'tools3d'],
    install_requires='openexr>=1.3.0',
    dependency_links=['https://github.com/jamesbowman/openexrpython/tarball/master#egg=openexr-1.3.0'],
)
