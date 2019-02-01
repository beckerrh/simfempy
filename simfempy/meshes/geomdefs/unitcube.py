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
    def define(self, h=1.):
        self.reset()
        a = 1
        p = self.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=-a, lcar=h)
        self.add_physical_surface(p.surface, label=100)
        axis = [0, 0, 2*a]
        top, vol, ext = self.extrude(p.surface, axis)
        # print ('vol', vars(vol))
        # print ('top', vars(top))
        # print ('top.id', top.id)
        # print ('ext[0]', vars(ext[0]))
        self.add_physical_surface(top, label=105)
        self.add_physical_surface(ext[0], label=101)
        self.add_physical_surface(ext[1], label=102)
        self.add_physical_surface(ext[2], label=103)
        self.add_physical_surface(ext[3], label=104)
        self.add_physical_volume(vol, label=10)
        return self

# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import pygmsh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = Unitcube(h=2)
    meshdata = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshdata)
    mesh.plotWithBoundaries()
    plt.show()
    mesh.plot(localnumbering=True)
    plt.show()
