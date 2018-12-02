# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse as sparse
try:
    from fempy.meshes.trianglemesh import TriangleMesh
except ModuleNotFoundError:
    from ..meshes.trianglemesh import TriangleMesh


#=================================================================#
class FemP12D(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)
        self.locmatmass = np.zeros((3,3))
        self.locmatlap = np.zeros((3,3))
    def setMesh(self, mesh):
        assert isinstance(mesh, TriangleMesh)
        self.mesh = mesh
        self.xedges = self.mesh.x[self.mesh.edges].mean(axis=1)
        self.yedges = self.mesh.y[self.mesh.edges].mean(axis=1)
    def phi(self, ic, x, y, grad):
        return 1./3. + np.dot(grad, np.array([x-self.mesh.centersx[ic], y-self.mesh.centersy[ic]]))
    def elementMassMatrix(self, ic):
        scale = self.mesh.area[ic]/12
        for ii in range(3):
            for jj in range(3):
                self.locmatmass[ii,jj] = scale
        for ii in range(3):
            self.locmatmass[ii, ii] *= 2
        return self.locmatmass
    def elementLaplaceMatrix(self, ic):
        scale = self.mesh.area[ic]
        grads = self.grad(ic)
        for ii in range(3):
            for jj in range(3):
                self.locmatlap[ii,jj] = np.dot(grads[ii],grads[jj])*scale
        return self.locmatlap
    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.edgesOfCell[ic,:]]
        grads = 0.5*normals/self.mesh.area[ic]
        chsg =  (ic == self.mesh.cellsOfEdge[self.mesh.edgesOfCell[ic,:],0])
        grads[chsg] *= -1.
        return grads
    def testgrad(self):
        for ic in range(fem.mesh.ncells):
            grads = fem.grad(ic)
            for ii in range(3):
                x = self.mesh.x[self.mesh.triangles[ic,ii]]
                y = self.mesh.y[self.mesh.triangles[ic,ii]]
                for jj in range(3):
                    phi = self.phi(ic, x, y, grads[jj])
                    if ii == jj:
                        test = np.abs(phi-1.0)
                        if test > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            print('x,y', x, y)
                            print('x-xc,y-yc', x-self.mesh.centersx[ic], y-self.mesh.centersy[ic])
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic,ii,jj, test))
                    else:
                        test = np.abs(phi)
                        if np.abs(phi) > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic,ii,jj, test))

    def massMatrix(self):
        nnodes, ncells = self.mesh.nnodes, self.mesh.ncells
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        nlocal = 9
        index = np.zeros(nlocal*ncells, dtype=int)
        jndex = np.zeros(nlocal*ncells, dtype=int)
        A = np.zeros(nlocal*ncells, dtype=np.float64)
        count = 0
        for ic in range(ncells):
            mass = self.elementMassMatrix(ic)
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = simps[ic, ii]
                    jndex[count+3*ii+jj] = simps[ic, jj]
                    A[count + 3 * ii + jj] += mass[ii,jj]
            count += nlocal
        return sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes)).tocsr()


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = TriangleMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemP12D(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
