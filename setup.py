#!/usr/bin/env python

import shutil
from distutils.core import setup

from setuptools import find_packages


setup(name='RNAformer',
      version='1.0.1',
      packages=find_packages(where='RNAformer'),
      package_dir={'': 'RNAformer'},
      dependency_links=[],
      )
