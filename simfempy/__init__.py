from . import applications, fems, meshes, solvers, tools

from .__about__ import __author__, __author_email__, __version__, __website__
__all__ = [
    "__version__",
    "__author__",
    "__author_email__",
    "__website__",
]
print(f"############### version is {__version__}")
import pygmsh
print(f"############### pygmsh version is {pygmsh.__version__}")
# if pygmsh.__version__ >= 7:
#     raise ImportError(f"Needs pygmsh version 6, installed is {pygmsh.__version__} ")