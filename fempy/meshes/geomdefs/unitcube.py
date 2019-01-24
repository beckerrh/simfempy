# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import pygmsh

# ------------------------------------- #
def define_geometry(h=1.0):
    geometry = pygmsh.built_in.Geometry()
    a = 1
    p = geometry.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=-a, lcar=h)
    geometry.add_physical_surface(p.surface, label=100)
    axis = [0, 0, 2*a]
    top, vol, ext = geometry.extrude(p.surface, axis)
    # print ('vol', vars(vol))
    # print ('top', vars(top))
    # print ('top.id', top.id)
    # print ('ext[0]', vars(ext[0]))
    geometry.add_physical_surface(top, label=105)
    geometry.add_physical_surface(ext[0], label=101)
    geometry.add_physical_surface(ext[1], label=102)
    geometry.add_physical_surface(ext[2], label=103)
    geometry.add_physical_surface(ext[3], label=104)
    geometry.add_physical_volume(vol, label=10)
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
