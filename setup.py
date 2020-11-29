# -*- coding: utf-8 -*-
#
import os
from setuptools import setup, find_packages

base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "simfempy", "__about__.py"), "rb") as f:
    exec(f.read(), about)
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simfempy",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=find_packages(),
    url=about["__website__"],
    license=about["__license__"],
    description="A small package for fem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms="any",
    classifiers=[
        about["__license__"],
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.6',
)