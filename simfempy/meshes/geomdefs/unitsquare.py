# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import pygmsh

# ------------------------------------- #
def define_geometry(h=1.):
    geometry = pygmsh.built_in.Geometry()
    a = 1.0
    p = geometry.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=0, lcar=h)
    geometry.add_physical_surface(p.surface, label=100)
    for i in range(4): geometry.add_physical_line(p.line_loop.lines[i], label=1000+i)
    return geometry

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import plotmesh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = define_geometry(h=2)
    meshdata = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshdata)
    mesh.plotWithBoundaries()
    plt.show()
    mesh.plot(localnumbering=True)
    plt.show()
