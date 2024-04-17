#!/usr/bin/env python

import shutil
from distutils.core import setup

from setuptools import find_packages


setup(name='RNAformer',
      version='1.1.0',
      packages=find_packages(),
      include_package_data=True,
      package_dir={'': '.'},
      dependency_links=[],
      )
