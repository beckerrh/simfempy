# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
try:
    from simfempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class FemD0(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
    def computeErrorL2(self, solex, uh):
        x, y, z = self.mesh.pointsc[:,0], self.mesh.pointsc[:,1], self.mesh.pointsc[:,2]
        e = uh - solex(x, y, z)
        return np.sqrt( np.dot(e, self.mesh.dV*e) ), e


#=================================================================#
if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemD0(trimesh)
    raise NotImplementedError
