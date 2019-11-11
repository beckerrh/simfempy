# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

try:
    from . import geometry
except:
    import geometry

# ------------------------------------- #
class Unitcube(geometry.Geometry):
    def __init__(self, **kwargs):
        self.x, self.y, self.z = [-1, 1], [-1, 1], [-1, 1]
        if 'x' in kwargs: self.x = kwargs['x']
        if 'y' in kwargs: self.x = kwargs['y']
        if 'z' in kwargs: self.x = kwargs['z']
        h = None
        if 'h' in kwargs: h = kwargs['h']
        super().__init__(h)
    def define(self, h=1.):
        self.reset()
        x, y, z = self.x, self.y, self.z
        p = self.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
        self.add_physical(p.surface, label=100)
        axis = [0, 0, z[1]-z[0]]
        top, vol, ext = self.extrude(p.surface, axis)
        self.add_physical(top, label=105)
        self.add_physical(ext[0], label=101)
        self.add_physical(ext[1], label=102)
        self.add_physical(ext[2], label=103)
        self.add_physical(ext[3], label=104)
        self.add_physical(vol, label=10)
        return self

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Unitcube(h=2)
    mesh = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(mesh=mesh)
    mesh.plotWithBoundaries()
    plt.show()
