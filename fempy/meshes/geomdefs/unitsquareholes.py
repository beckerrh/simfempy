# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import pygmsh
import numpy as np

# ------------------------------------- #
def define_geometry(h=0.1, rect = [-1, 1, -1, 1], holes = [ \
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5] ]]):
    geometry = pygmsh.built_in.Geometry()
    xholes = []
    for hole in holes:
        xholes.append(np.insert(np.array(hole), 2, 0, axis=1))
    holes=[]
    for i,xhole in enumerate(xholes):
        holes.append(geometry.add_polygon(X=xhole, lcar=h))
        geometry.add_physical_surface(holes[i].surface, label=200+i)

    # outer rectangle
    p1 = geometry.add_rectangle(rect[0], rect[1], rect[2], rect[3], 0, lcar=h, holes=holes)
    geometry.add_physical_surface(p1.surface, label=100)
    for i in range(4): geometry.add_physical_line(p1.line_loop.lines[i], label=11*(1+i))
    return geometry

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import plotmesh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = define_geometry(h=1.0)
    meshdata = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshdata)
    plotmesh.meshWithBoundaries(mesh)
    plt.show()
    plotmesh.meshWithData(mesh, cell_data={'labels':meshdata[3]['triangle']['gmsh:physical']})
    plt.show()
