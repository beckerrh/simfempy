# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
try:
    from . import geometry
except:
    import geometry
import numpy as np

# ------------------------------------- #
class Equilateral(geometry.Geometry):
    def define(self, h=1.):
        self.reset()
        a = 1.0
        X = []
        X.append([-0.5*a, 0, 0])
        X.append([0, -0.5*np.sqrt(3)*a, 0])
        X.append([0.5*a, 0, 0])
        X.append([0, 0.5*np.sqrt(3)*a, 0])
        p = self.add_polygon(X=X, lcar = h)
        self.add_physical(p.surface, label=100)
        for i in range(4): self.add_physical(p.line_loop.lines[i], label=1000 + i)

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Equilateral(h=2)
    meshmesh = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshmesh)
    mesh.plotWithBoundaries()
    plt.show()
    mesh.plot(localnumbering=True)
    plt.show()
