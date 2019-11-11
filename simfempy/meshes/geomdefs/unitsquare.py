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
class Unitsquare(geometry.Geometry):
    def define(self, h=1.):
        print(f"h ={h}")
        self.reset()
        a = 1.0
        p = self.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=0, lcar=h)
        self.add_physical(p.surface, label=100)
        for i in range(4): self.add_physical(p.line_loop.lines[i], label=1000 + i)

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Unitsquare(h=2)
    mesh = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(mesh=mesh)
    mesh.plotWithBoundaries()
    plt.show()
    mesh.plot(localnumbering=True)
    plt.show()
