# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import numpy.linalg as linalg
from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class Fem(object):
    def __init__(self, mesh=None):
        if mesh is not None: self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
    def downWind(self, v, method='supg'):
        # v is supposed RT0
        dim, ncells, fofc, sigma = self.mesh.dimension, self.mesh.ncells, self.mesh.facesOfCells, self.mesh.sigma
        normals, dV = self.mesh.normals, self.mesh.dV
        if method=='supg':
            dS = linalg.norm(normals[fofc],axis=2)
            print(f"{dS.shape=}")
            vs = v[fofc]*sigma*dS/dV[:,np.newaxis]/dim
            vp = np.maximum(vs, 0)
            ips = vp.argmax(axis=1)
            # print(f"{ips=}")
            vps = np.choose(ips, vp.T)
            # print(f"{vps=}")
            delta = 1/vps
            print(f"{delta.shape=} {sigma.shape} {v[fofc].shape}")
            if not np.all(delta > 0): raise ValueError(f"{delta=}\n{vp[ips]=}")
            lamd = (np.ones(ncells)[:,np.newaxis] - delta[:,np.newaxis]*vs)/(dim+1)
            if not np.allclose(lamd.sum(axis=1),1):
                raise ValueError(f"{lamd=}\n{(v[fofc]*sigma).sum(axis=1)}")
        elif method=='supg2':
            # vp = np.maximum(v[fofc]*sigma, 0)
            # vps = vp.sum(axis=1)
            # if not np.all(vps > 0): raise ValueError(f"{vps=}\n{vp=}")
            # vp /= vps[:,np.newaxis]
            # lamd = (np.ones(ncells)[:,np.newaxis] - vp)/dim
            vm = np.minimum(v[fofc]*sigma, 0)
            vms = vm.sum(axis=1)
            if not np.all(vms < 0): raise ValueError(f"{vms=}\n{vm=}")
            vm /= vms[:,np.newaxis]
            lamd = vm
        # print(f"{v[fofc].shape} {lamd2.shape=}")
        points, simplices = self.mesh.points, self.mesh.simplices
        xd = np.einsum('nji,nj -> ni', points[simplices], lamd)
        # if not np.allclose(lamd, lamd2): print(f"{lamd=} {lamd2=}")
        return xd, lamd




# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
