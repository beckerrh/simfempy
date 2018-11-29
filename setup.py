# -*- coding: utf-8 -*-
#
import os

from setuptools import setup, find_packages

base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "pygmsh", "__about__.py"), "rb") as f:
    exec(f.read(), about)


setup(
    name="pygmsh",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=find_packages(),
    url=about["__website__"],
    license=about["__license__"],
    platforms="any",
    classifiers=[
        about["__status__"],
        about["__license__"],
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)