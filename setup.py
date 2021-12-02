# -*- coding: utf-8 -*-
#
from setuptools import setup, find_packages

VERSION = "2.0.25"

with open("simfempy/examples/heat.py", "r") as heat:
    example = heat.read()
with open("README.md", "w") as readme:
    readme.write("SIMple Finite Element Methods in PYthon\n\n```python\n")
    readme.write(example)
    readme.write("\n```\n")
with open("README.md", "r") as readme:
    long_description = readme.read()
setup(
    name="simfempy",
    version=VERSION,
    author="Roland Becker",
    author_email="beckerrolandh@gmail.com",
    packages=find_packages(),
    url="https://github.com/beckerrh/simfempy",
    license="License :: OSI Approved :: MIT License",
    description="A small package for fem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms="any",
    install_requires=['gmsh', 'pygmsh', 'meshio', 'scipy', 'sympy', 'matplotlib'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.8',
)