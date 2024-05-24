#!/usr/bin/env python
from setuptools import setup, find_packages


setup(name='RNAformer',
      version='0.0.1',
      packages=find_packages(),
      include_package_data=True,
      package_dir={'': '.'},
      install_requires=[
            "torch",
            "torchvision",
            "torchaudio",
            "tqdm",
            "pyyaml",
            "pyaml",
            "numpy",
            "packaging",
            "wheel",
            "tabulate",
            "scipy",
            "pandas==2.0.2",
            "scikit-learn==1.3.0",
            "matplotlib==3.7.2",
            "polars",
            "loralib==0.1.2",
            "tensorboard==2.13.0",
            "transformers",
            "datasets==2.13.1",
            "pytorch-lightning==2.0.4",
            "deepspeed==0.9.5",
            "rotary-embedding-torch==0.5.3",
      ],
      )
