from distutils.core import setup

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
    scripts = ['scripts/loki.queryRingIndices']
    )
