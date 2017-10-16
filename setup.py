# Set this to `False` for production use.
# Better keep this `False` in git-committed version.
debug = False

from distutils.version import LooseVersion, StrictVersion
import sys

py_version = sys.version.split()[0]
if py_version < LooseVersion('3.5.3'):
    msg = ('Python 3.5.3 or above is required. You have version %s.\n'
           'If the required version is already installed, '
           'check whether the command is called `python3` instead of `python`.') % py_version
    sys.exit(msg)

# Now that Python 3 is confirmed, we can use `importlib`.
from importlib import import_module


def import_strict_module(module, version):
    try:
        mm = import_module(module)
    except:
        msg = '%s version %s is required. Please `pip install %s==%s` and try again.' \
                % (module, version, module, version)
        sys.exit(msg)
    ver = mm.__version__
    if ver != StrictVersion(version):
        msg = '%s version %s is required; you have version %s. Please `pip install %s==%s` and try again.' \
                % (module, version, ver, module, version)
        sys.exit(msg)
    return mm


def import_min_module(module, version):
    try:
        mm = import_module(module)
    except:
        msg = '%s version %s or above is required. Please `pip install -U %s>=%s` and try again.' \
                % (module, version, module, version)
        sys.exit(msg)
    ver = mm.__version__
    if ver < LooseVersion(version):
        msg = '%s version %s or above is required; you have version %s. Please `pip install -U %s>=%s` and try again.' \
                % (module, version, ver, module, version)
        sys.exit(msg)
    return mm


setuptools = import_min_module('setuptools', '22.0.0')
from setuptools import setup, Extension

Cython = import_strict_module('Cython', '0.25.2')
from Cython.Build import cythonize

numpy = import_min_module('numpy', '1.12.0')
numpy_include_dir = numpy.get_include()


cy_options = {
    'annotate': True,
    'compiler_directives': {
        'profile': debug,
        'linetrace': debug,
        'boundscheck': debug,
        'wraparound': debug,
        'initializedcheck': debug,
        'language_level': 3,
    },
}


extensions = [
    Extension('_transformer', ['datamill/_transformer.pyx'],
              define_macros=[('CYTHON_TRACE', '1' if debug else '0')],
              extra_compile_args=['-O3'],
             ),
]


setup(
    name='datamill',
    version='0.1',
    install_requires=[
        'pytest>=3.0.1',
    ],
    packages=['datamill',],
    ext_package='datamill',
    ext_modules=cythonize(extensions, **cy_options),
)

