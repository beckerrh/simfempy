# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
try:
    from fempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh


#=================================================================#
class FemRT0(object):
    """
    on suppose que  self.mesh.edgesOfCell[ic, kk] et oppose à elem[ic,kk] !!!
    """
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)

    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        facesofcells = self.mesh.facesOfCells
        self.rows = np.repeat(facesofcells, self.nloc).flatten()
        self.cols = np.tile(facesofcells, self.nloc).flatten()
        self.Mtocell = self.toCellMatrix()
        self.pointsf = self.mesh.points[self.mesh.faces].mean(axis=1)


    def toCellMatrix(self):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        dim, dV, nloc, p, pc, simp = self.mesh.dimension, self.mesh.dV, self.nloc, self.mesh.points, self.mesh.pointsc, self.mesh.simplices
        dS = sigma * linalg.norm(normals[facesofcells], axis=2)/dim
        # ps = p[simp]
        # print("ps.shape", ps.shape)
        # print("ps", ps)
        ps = p[simp][:,:,:dim]
        # print("ps.shape", ps.shape)
        # print("ps", ps)
        ps2 = np.transpose(ps, axes=(2,0,1))
        # print("ps2.shape", ps2.shape)
        # print("ps2", ps2)
        pc2 = np.repeat(pc[:,:dim].T[:, :, np.newaxis], nloc, axis=2)
        # print("pc2.shape", pc2.shape)
        # print("pc2", pc2)
        pd = pc2 -ps2
        rows = np.repeat((np.repeat(dim * np.arange(ncells), dim).reshape(ncells,dim) + np.arange(dim)).swapaxes(1,0),nloc)
        cols = np.tile(facesofcells.ravel(), dim)
        mat = np.einsum('ni, jni, n->jni', dS, pd, 1/dV)
        # print("rows.shape", rows.shape, "cols.shape", cols.shape)
        # print("pd.shape", pd.shape)
        # print("dS.shape", dS.shape)
        # print("mat.shape", mat.shape)

        # mat2 = np.zeros(shape=(dim,ncells,nloc))
        # rows2 = np.zeros(shape=(dim,ncells,nloc), dtype=int)
        # cols2 = np.zeros(shape=(dim,ncells,nloc), dtype=int)
        # for ic in range(ncells):
        #     for ii in range(nloc):
        #         for idim in range(dim):
        #             mat2[idim, ic, ii] += (pc[ic][idim] - p[simp[ic,ii]][idim])/dV[ic]*dS[ic,ii]
        #             rows2[idim, ic, ii] = ic*dim + idim
        #             cols2[idim, ic, ii] = facesofcells[ic,ii]
        #
        # print("cols", cols.flatten())
        # print("cols2", cols2.flatten())
        # assert np.allclose(cols.flatten(), cols2.flatten())
        # print("rows", rows.flatten())
        # print("rows2", rows2.flatten())
        # assert np.allclose(rows.flatten(), rows2.flatten())
        # assert np.allclose(mat, mat2)

        return  sparse.coo_matrix((mat.flatten(), (rows.flatten(), cols.flatten())), shape=(dim*ncells, nfaces))

    def toCell(self, v):
        return self.Mtocell.dot(v)

    def constructMass(self):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        dim, dV, nloc, p, pc, simp = self.mesh.dimension, self.mesh.dV, self.nloc, self.mesh.points, self.mesh.pointsc, self.mesh.simplices
        scalea = 1 / dim / dim / (dim + 2) / (dim + 1)
        scaleb = 1 / dim / dim / (dim + 2) * (dim + 1)
        scalec = 1 / dim / dim
        dS = sigma * linalg.norm(normals[facesofcells], axis=2)
        # print("p[simp].shape", p[simp].shape)
        x1 = scalea *np.einsum('nij,nij->n', p[simp], p[simp]) + scaleb* np.einsum('ni,ni->n', pc, pc)
        mat = np.einsum('ni,nj, n->nij', dS, dS, x1)
        x2 = scalec *np.einsum('nik,njk->nij', p[simp], p[simp])
        mat += np.einsum('ni,nj,nij->nij', dS, dS, x2)
        x3 = - scalec * np.einsum('nik,nk->ni', p[simp], pc)
        mat += np.einsum('ni,nj,ni->nij', dS, dS, x3)
        mat += np.einsum('ni,nj,nj->nij', dS, dS, x3)
        mat = np.einsum("nij, n -> nij", mat, 1/dV)
        return sparse.coo_matrix((mat.flatten(), (self.rows, self.cols)), shape=(nfaces, nfaces))

    def constructDiv(self):
        ncells, nfaces, normals, sigma, facesofcells = self.mesh.ncells, self.mesh.nfaces, self.mesh.normals, self.mesh.sigma, self.mesh.facesOfCells
        rows = np.repeat(np.arange(ncells), self.nloc)
        cols = facesofcells.flatten()
        mat =  (sigma*linalg.norm(normals[facesofcells],axis=2)).flatten()
        return  sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, nfaces))

