# -*- coding: utf-8 -*-
#
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

print(f"{find_packages()=}")
setup(
    name="simfempy",
    version="2.0.12",
    author="Roland Becker",
    author_email="beckerrolandh@gmail.com",
    packages=find_packages(),
    url="https://github.com/beckerrh/simfempy",
    license="License :: OSI Approved :: MIT License",
    description="A small package for fem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms="any",
    install_requires=['gmsh', 'pygmsh', 'meshio', 'scipy', 'sympy', 'pyamg', 'matplotlib'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.8',
)