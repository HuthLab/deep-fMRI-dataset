#/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
requirements = [
]
setup(
    name="encoding",
    version="0.1.0",
    description="Code for fitting fMRI encoding",
    author="Amanda LeBel",
    author_email="amanda_lebel@berkeley.edu",
    packages=[
        'encoding',
    ],
    package_dir={'encoding':
                 'encoding'},
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
    ],
)
