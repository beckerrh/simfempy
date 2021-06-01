# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import numpy.linalg as linalg
from simfempy import fems
from simfempy.meshes.simplexmesh import SimplexMesh
import scipy.sparse as sparse


#=================================================================#
class Fem(object):
    def __repr__(self):
        repr = f"{self.__class__.__name__}"
        return repr
    def __init__(self, **kwargs):
        mesh = kwargs.get('mesh', None)
        if mesh is not None: self.setMesh(mesh)
    def setMesh(self, mesh, innersides=False):
        self.mesh = mesh
        self.nloc = self.nlocal()
        if innersides: self.mesh.constructInnerFaces()
    def computeStencilCell(self, dofspercell):
        self.cols = np.tile(dofspercell, self.nloc).ravel()
        self.rows = np.repeat(dofspercell, self.nloc).ravel()
        #Alternative
        # self.rows = dofspercell.repeat(self.nloc).reshape(self.mesh.ncells, self.nloc, self.nloc)
        # self.cols = self.rows.swapaxes(1, 2)
        # self.cols = self.cols.reshape(-1)
        # self.rows = self.rows.reshape(-1)

    # def computeStencilInnerSidesCell(self, dofspercell):
    #     nloc, faces, cellsOfFaces = self.nloc, self.mesh.faces, self.mesh.cellsOfFaces
    #     # print(f"{faces=}")
    #     # print(f"{cellsOfFaces=}")
    #     innerfaces = cellsOfFaces[:,1]>=0
    #     cellsOfInteriorFaces= cellsOfFaces[innerfaces]
    #     self.cellsOfInteriorFaces = cellsOfInteriorFaces
    #     self.innerfaces = innerfaces
    #     return
    #     # print(f"{innerfaces=}")
    #     print(f"{cellsOfInteriorFaces=}")
    #     raise NotImplementedError(f"no")
    #     ncells, nloc = dofspercell.shape[0], dofspercell.shape[1]
    #     print(f"{ncells=} {nloc=}")
    #     print(f"{dofspercell[cellsOfInteriorFaces,:].shape=}")
    #     rows = dofspercell[cellsOfInteriorFaces,:].repeat(nloc)
    #     cols = np.tile(dofspercell[cellsOfInteriorFaces,:],nloc)
    #     print(f"{rows=}")
    #     print(f"{cols=}")
    def interpolateCell(self, f):
        if isinstance(f, dict):
            b = np.zeros(self.mesh.ncells)
            for label, fct in f.items():
                if fct is None: continue
                cells = self.mesh.cellsoflabel[label]
                xc, yc, zc = self.mesh.pointsc[cells].T
                b[cells] = fct(xc, yc, zc)
            return b
        else:
            xc, yc, zc = self.mesh.pointsc.T
            return f(xc, yc, zc)
    def computeMatrixDiffusion(self, coeff):
        ndofs = self.nunknowns()
        # matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        # matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        # matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        # mat = ( (matxx+matyy+matzz).T*self.mesh.dV*coeff).T.ravel()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        mat = np.einsum('n,nil,njl->nij', self.mesh.dV*coeff, cellgrads, cellgrads).ravel()
        return sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(ndofs, ndofs)).tocsr()
    def computeFormDiffusion(self, du, u, coeff):
        doc = self.dofspercell()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        r = np.einsum('n,nil,njl,nj->ni', self.mesh.dV*coeff, cellgrads, cellgrads, u[doc])
        np.add.at(du, doc, r)

    def computeMatrixLps(self, betaC):
        dimension, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
        nloc, dofspercell = self.nlocal(), self.dofspercell()
        ci = self.mesh.cellsOfInteriorFaces
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci[:,0]]+ dV[ci[:,1]])
        betan = 0.5*(np.linalg.norm(betaC[ci[:,0]],axis=1)+ np.linalg.norm(betaC[ci[:,1]],axis=1))
        scale *= 0.01*dS*betan
        cg0 = self.cellgrads[ci[:,0], :, :]
        cg1 = self.cellgrads[ci[:,1], :, :]
        mat00 = np.einsum('nki,nli,n->nkl', cg0, cg0, scale)
        mat01 = np.einsum('nki,nli,n->nkl', cg0, cg1, -scale)
        mat10 = np.einsum('nki,nli,n->nkl', cg1, cg0, -scale)
        mat11 = np.einsum('nki,nli,n->nkl', cg1, cg1, scale)
        rows0 = dofspercell[ci[:,0],:].repeat(nloc)
        cols0 = np.tile(dofspercell[ci[:,0],:],nloc).reshape(-1)
        rows1 = dofspercell[ci[:,1],:].repeat(nloc)
        cols1 = np.tile(dofspercell[ci[:,1],:],nloc).reshape(-1)
        A00 = sparse.coo_matrix((mat00.reshape(-1), (rows0, cols0)), shape=(ndofs, ndofs))
        A01 = sparse.coo_matrix((mat01.reshape(-1), (rows0, cols1)), shape=(ndofs, ndofs))
        A10 = sparse.coo_matrix((mat10.reshape(-1), (rows1, cols0)), shape=(ndofs, ndofs))
        A11 = sparse.coo_matrix((mat11.reshape(-1), (rows1, cols1)), shape=(ndofs, ndofs))
        return A00+A01+A10+A11

    def computeFormConvection(self, du, u, data, method):
        if method[:4] == 'supg':
            self.computeFormTransportSupg(du, u, data, method)
        elif method == 'upwalg':
            self.computeFormTransportUpwindAlg(du, u, data)
        elif method[:3] == 'upw':
            self.computeFormTransportUpwind(du, u, data, method)
        elif method == 'lps':
            self.computeFormTransportLps(du, u, data)
        else:
            raise NotImplementedError(f"{method=}")
    def computeMatrixConvection(self, data, method):
        if method[:4] == 'supg':
            return self.computeMatrixTransportSupg(data, method)
        elif method == 'upwalg':
            return self.computeMatrixTransportUpwindAlg(data)
        elif method[:3] == 'upw':
            return self.computeMatrixTransportUpwind(data, method)
        elif method == 'lps':
            return self.computeMatrixTransportLps(data)
        else:
            raise NotImplementedError(f"{method=}")

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
