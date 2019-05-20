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
class Unitsquareholes(geometry.Geometry):
    def define(self, h=0.1, rect = [-1, 1, -1, 1], holes = [ \
            [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5] ]]):
        self.reset()
        xholes = []
        for hole in holes:
            xholes.append(np.insert(np.array(hole), 2, 0, axis=1))
        holes=[]
        for i,xhole in enumerate(xholes):
            holes.append(self.add_polygon(X=xhole, lcar=h))
            self.add_physical(holes[i].surface, label=200+i)
        # outer rectangle
        p1 = self.add_rectangle(rect[0], rect[1], rect[2], rect[3], 0, lcar=h, holes=holes)
        self.add_physical(p1.surface, label=100)
        for i in range(4): self.add_physical(p1.line_loop.lines[i], label=1000+i)
        return self

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Unitsquareholes(h=1)
    meshmesh = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshdata)
    mesh.plotWithBoundaries()
    plt.show()
    mesh.plot(localnumbering=True)
    plt.show()
