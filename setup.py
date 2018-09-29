from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Build import cythonize

    def get_numpy_include():
        """ 
        Obtain the numpy include directory. This logic works across numpy versions.
        """
        try:
            numpy_include = numpy.get_include()
        except AttributeError:
            numpy_include = numpy.get_numpy_include()
        return numpy_include


    corr = Extension('loki.corr',
                 sources=['src/corr/correlate.pyx', 'src/corr/corr.cpp'],
                 extra_compile_args = ['-O3', '-fPIC', '-Wall'],
                 runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                 extra_link_args = ['-lstdc++', '-lm'],
                 include_dirs = [get_numpy_include(), 'src/corr'],
                 language='c++') 
    mod = cythonize( [corr]) 

except ImportError:
    mod = []


setup(
    name='Loki',
    version='1.0',
    description='Useful tools for analyzing X-ray'\
                +' diffraction images containing ring patterns',
    author='Derek Anthony Mendez Jr.',
    author_email='dermendarko@gmail.com',
    url='https://github.com/dermen/loki',
    packages=['loki', 'loki.RingData', 'loki.utils'],
    package_dir={'loki':'src'},
    scripts = ['scripts/loki.queryRingIndices'],
    ext_modules=mod) 
