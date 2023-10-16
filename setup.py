from pathlib import Path
from setuptools import setup


skylibs_module = {}
root = Path(__file__).resolve().parent
with open(str(root / "skylibs/__init__.py")) as fhdl:
    exec(fhdl.read(), skylibs_module)


setup(
    name='skylibs',
    description=('Tools to read, write, perform projections and handle LDR/HDR environment maps (IBL).'),
    author='Yannick Hold',
    author_email='yannickhold@gmail.com',
    license="LGPLv3",
    url='https://github.com/soravux/skylibs',
    version=skylibs_module['__version__'],
    packages=['ezexr', 'envmap', 'hdrio', 'hdrtools', 'hdrtools/tonemapping', 'skydb', 'tools3d', 'tools3d/warping_operator', 'skylibs'],
    include_package_data=True,
    install_requires=['imageio>=1.6', 'tqdm', 'numpy', 'scipy', 'scikit-image>=0.19', 'pysolar'],
)
