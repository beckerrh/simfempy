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
class Unitline(geometry.Geometry):
    def define(self, h=1.):
        self.reset()
        p0 = self.add_point([0,0,0], lcar=h)
        p1 = self.add_point([1,0,0], lcar=h)
        p = self.add_line(p0, p1)
        self.add_physical(p0, label=10000)
        self.add_physical(p1, label=10001)
        self.add_physical(p, label=1000)

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Unitline(h=0.2)
    mesh = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(mesh=mesh)
    mesh.plotWithBoundaries()
    plt.show()
