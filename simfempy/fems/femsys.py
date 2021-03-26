# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

#=================================================================#
class Femsys():
    def __init__(self, fem, ncomp, mesh=None):
        self.ncomp = ncomp
        self.fem = fem
    def nlocal(self): return self.fem.nlocal()
    def nunknowns(self): return self.fem.nunknowns()
    def dofspercell(self): return self.fem.dofspercell()
    def setMesh(self, mesh):
        self.mesh = mesh
        self.fem.setMesh(mesh)
        ncomp, nloc, ncells = self.ncomp, self.fem.nloc, self.mesh.ncells
        dofs = self.fem.dofspercell()
        nlocncomp = ncomp * nloc
        self.rowssys = np.repeat(ncomp * dofs, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp, dtype="uint")
        self.rowssys = self.rowssys.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.colssys = self.rowssys.swapaxes(1, 2)
        self.colssys = self.colssys.reshape(-1)
        self.rowssys = self.rowssys.reshape(-1)
    def prepareBoundary(self, colorsdirichlet, colorsflux=[]):
        self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
    def computeErrorL2(self, solex, uh):
        eall, ecall = [], []
        for icomp in range(self.ncomp):
            e, ec = self.fem.computeErrorL2(solex[icomp], uh[icomp::self.ncomp])
            eall.append(e)
            ecall.append(ec)
        return eall, ecall
    def computeBdryMean(self, u, colors):
        all = []
        for icomp in range(self.ncomp):
            a = self.fem.computeBdryMean(u[icomp::self.ncomp], colors)
            all.append(a)
        return all
    def computeMatrixLps(self):
        ncomp = self.ncomp
        dimension, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
        nloc, dofspercell, nall = self.nlocal(), self.dofspercell(), ncomp*ndofs
        ci = self.mesh.cellsOfInteriorFaces
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci[:,0]]+ dV[ci[:,1]])
        scale *= 0.0001*dS
        cg0 = self.fem.cellgrads[ci[:,0], :, :]
        cg1 = self.fem.cellgrads[ci[:,1], :, :]
        mat00 = np.einsum('nki,nli,n->nkl', cg0, cg0, scale)
        mat01 = np.einsum('nki,nli,n->nkl', cg0, cg1, -scale)
        mat10 = np.einsum('nki,nli,n->nkl', cg1, cg0, -scale)
        mat11 = np.einsum('nki,nli,n->nkl', cg1, cg1, scale)
        A = sparse.coo_matrix((nall, nall))
        for icomp in range(ncomp):
            d0 = ncomp*dofspercell[ci[:,0],:]+icomp
            d1 = ncomp*dofspercell[ci[:,1],:]+icomp
            rows0 = d0.repeat(nloc)
            cols0 = np.tile(d0,nloc).reshape(-1)
            rows1 = d1.repeat(nloc)
            cols1 = np.tile(d1,nloc).reshape(-1)
            A += sparse.coo_matrix((mat00.ravel(), (rows0, cols0)), shape=(nall, nall))
            A += sparse.coo_matrix((mat01.ravel(), (rows0, cols1)), shape=(nall, nall))
            A += sparse.coo_matrix((mat10.ravel(), (rows1, cols0)), shape=(nall, nall))
            A += sparse.coo_matrix((mat11.ravel(), (rows1, cols1)), shape=(nall, nall))
        return A

# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError(f"pas de test")
