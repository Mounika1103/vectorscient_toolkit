"""
"""
from setuptools import setup


classifiers = [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python',
    'Operating system :: MacOS :: MacOS X',
    'Operating system :: POSIX :: Linux',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: Other/Proprietary License'
    'Topic :: Utilities',
    'Topic :: Database',
    'Topic :: Scientific/Engineering'
]

with open("requirements.txt") as fp:
    required_packages = [l for l in fp.read().splitlines()
                         if l and not l.startswith("#")]

setup(name='vectorscient_toolkit',
      version='0.2.1',
      description='',
      long_description=__doc__,
      author='VectorScient',
      author_email='',
      url='https://github.com/Vectorscient/vectorscient_toolkit',
      packages=['vectorscient_toolkit',
                'vectorscient_toolkit.exceptions',
                'vectorscient_toolkit.resources',
                'vectorscient_toolkit.predict_ally',
                'vectorscient_toolkit.predict_ally.clustering',
                'vectorscient_toolkit.log',
                'vectorscient_toolkit.qfi',
                'vectorscient_toolkit.pdf',
                'vectorscient_toolkit.phrasely',
                'vectorscient_toolkit.sources',
                'vectorscient_toolkit.sources.utils',
                'vectorscient_toolkit.stats',
                'vectorscient_toolkit.transformation',
                'vectorscient_toolkit.tests',
                'vectorscient_toolkit.functional_tests'],

      package_data={
          'vectorscient_toolkit.resources': ['images/*.png', 'texts/*.html']
      },

      # There is a bug with numpy (not sure how to deal with it):
      # https://github.com/numpy/numpy/issues/2434
      
      install_requires=required_packages,
      # Solution use: '`pip install `pwd`'

      classifiers=classifiers)
