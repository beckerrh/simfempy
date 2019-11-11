# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
try:
    from . import geometry
except:
    import geometry

# ------------------------------------- #
class Backwardfacingstep3d(geometry.Geometry):
    def define(self, h=1.):
        self.reset()
        X = []
        X.append([-1.0,  1.0])
        X.append([-1.0,  0.0])
        X.append([ 0.0,  0.0])
        X.append([ 0.0, -1.0])
        X.append([ 3.0, -1.0])
        X.append([ 3.0,  1.0])
        p = self.add_polygon(X=np.insert(np.array(X), 2, -1.0, axis=1), lcar=h)
        self.add_physical(p.surface, label=100)
        axis = [0, 0, 2]
        top, vol, ext = self.extrude(p.surface, axis)
        next = len(ext)
        self.add_physical(top, label=101+next)
        for i in range(next):
            self.add_physical(ext[i], label=101+i)
        self.add_physical(vol, label=10)
        return self

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Backwardfacingstep3d(h=2)
    meshmesh = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshdata)
    mesh.plotWithBoundaries()
    plt.show()
    mesh.plot(localnumbering=True)
    plt.show()
