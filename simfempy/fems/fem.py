# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class Fem(object):
    def __init__(self, mesh=None):
        if mesh is not None: self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
    def downWind(self, v):
        # v is supposed RT0
        dim, ncells, fofc, sigma = self.mesh.dimension, self.mesh.ncells, self.mesh.facesOfCells, self.mesh.sigma
        xf = self.mesh.pointsf
        xd = np.zeros((ncells,3))
        for k in range(ncells):
            s = 0
            for i in range(dim+1):
                vn = v[fofc[k,i]]*sigma[k,i]
                if(vn>0): s += vn
            for i in range(dim+1):
                vn = v[fofc[k,i]]*sigma[k,i]
                if(vn<=0): continue
                xd[k,:] += xf[fofc[k,i],:]*vn/s
        return xd




# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
