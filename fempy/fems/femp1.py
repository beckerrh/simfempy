# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import time
import numpy as np
import scipy.sparse as sparse
try:
    from fempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class FemP1(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)
    def setMesh(self, mesh):
        # assert isinstance(mesh, SimplexMesh)
        self.mesh = mesh
        nloc = self.mesh.dimension+1
        self.locmatmass = np.zeros((nloc, nloc))
        self.locmatlap = np.zeros((nloc, nloc))
        self.nloc = nloc
        ncells, simps = self.mesh.ncells, self.mesh.simplices
        npc = simps.shape[1]
        self.cols = np.tile(simps, npc).flatten()
        self.rows = np.repeat(simps, npc).flatten()
        self.computeFemMatrices()
    def massMatrix(self):
        nnodes = self.mesh.nnodes
        return sparse.coo_matrix((self.mass, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()

    def assemble(self, k):
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        return ( (matxx+matyy+matzz).T*self.mesh.dV*k).T.flatten()

    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = 1/self.mesh.dimension
        sigma = np.array([ 1.0 - 2.0 * (cellsOfFaces[facesOfCells[ic,:], 0] == ic) for ic in range(ncells)])
        self.cellgrads = scale*(normals[facesOfCells].T * sigma.T / dV.T).T
        scalemass = 1 / self.nloc / (self.nloc+1);
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] *= 2
        self.mass = np.einsum('n,kl->nkl', dV, massloc).flatten()

        # test gradients
        # for ic in range(ncells):
        #     grad = self.grad(ic)
        #     grad2 = cellgrads[ic]
        #     if not np.all(grad==grad2):
        #         print("true grad=\n{}\nwrong grad=\n{}".format(grad,grad2))
        #         print("sigma", sigma[ic])
        #         print("normals", normals[edgesOfCell[ic,:]])
        #         chsg = (ic == self.mesh.cellsOfEdge[self.mesh.edgesOfCell[ic, :], 0])
        #         print("chsg", chsg)
        #         raise ValueError("wrong grad")
        # self.matxx = np.einsum('nk,nl->nkl', self.cellgrads[:,:,0], self.cellgrads[:,:,0])
        # self.matxy = np.einsum('nk,nl->nkl', self.cellgrads[:,:,0], self.cellgrads[:,:,1])
        # self.matyx = np.einsum('nk,nl->nkl', self.cellgrads[:,:,1], self.cellgrads[:,:,0])
        # self.matyy = np.einsum('nk,nl->nkl', self.cellgrads[:,:,1], self.cellgrads[:,:,1])
        # self.matxx =  (self.matxx.T*area).T
        # self.matxy =  (self.matxy.T*area).T
        # self.matyx =  (self.matyx.T*area).T
        # self.matyy =  (self.matyy.T*area).T
        # print("self.matxx", self.matxx.shape)
        # #test laplace
        # for ic in range(ncells):
        #     A = self.elementLaplaceMatrix(ic)
        #     A2 = self.matxx[ic] + self.matyy[ic]
        #     if not np.allclose(A, A2):
        #         msg = "wrong element={} matrix\ngood={}\nwrong={}".format(ic,A,A2)
        #         raise ValueError(msg)
    def phi(self, ic, x, y, grad):
        return 1./3. + np.dot(grad, np.array([x-self.mesh.centersx[ic], y-self.mesh.centersy[ic]]))
    def elementMassMatrix(self, ic):
        nloc = self.mesh.dimension+1
        scale = self.mesh.dx[ic]/nloc/(nloc+1)
        for ii in range(nloc):
            for jj in range(nloc):
                self.locmatmass[ii,jj] = scale
        for ii in range(nloc):
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
        # print("### chsg", chsg, "normals", normals)
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

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = TriangleMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemP12D(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
