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
    def __repr__(self):
        repr = f"{self.__class__.__name__}:{self.fem.__class__.__name__}"
        return repr
    def __init__(self, fem, ncomp, mesh=None):
        self.ncomp = ncomp
        self.fem = fem
        if mesh: self.setMesh(mesh)
    def nlocal(self): return self.fem.nlocal()
    def nunknowns(self): return self.fem.nunknowns()
    def dofspercell(self): return self.fem.dofspercell()
    def setMesh(self, mesh):
        self.mesh = mesh
        self.fem.setMesh(mesh)
        ncomp, nloc, ncells = self.ncomp, self.fem.nloc, self.mesh.ncells
        dofs = self.fem.dofspercell()
        nlocncomp = ncomp * nloc
        self.rowssys = np.repeat(ncomp * dofs, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp, dtype=np.uint32)
        self.rowssys = self.rowssys.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.colssys = self.rowssys.swapaxes(1, 2)
        self.colssys = self.colssys.reshape(-1)
        self.rowssys = self.rowssys.reshape(-1)
    def prepareBoundary(self, colorsdirichlet, colorsflux=[]):
        return self.fem.prepareBoundary(colorsdirichlet, colorsflux)
    def computeRhsCells(self, b, rhs):
        rhsall = self.fem.interpolate(rhs)
        for i in range(self.ncomp):
            # print(f"{i=} {rhsall[i]=}")
            self.fem.massDot(b[i::self.ncomp], rhsall[i])
        return b
    def interpolate(self, rhs): return self.fem.interpolate(rhs).ravel()
    def massDot(self, b, f, coeff=1):
        if b.shape != f.shape:
            raise ValueError(f"{b.shape=} {f.shape=}")
        for i in range(self.ncomp):
            self.fem.massDot(b[i::self.ncomp], f[i::self.ncomp], coeff=coeff)
        return b
    def vectorBoundaryStrongZero(self, du, bdrydata):
        for i in range(self.ncomp): self.fem.vectorBoundaryStrongZero(du[i::self.ncomp], bdrydata)
        return du
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
    def getPointData(self, u):
        return {f"u_{i}":self.fem.tonode(u[i::self.ncomp]) for i in range(self.ncomp)}
    def matrix2systemdiagonal(self, A, ncomp):
        A = A.tocoo()
        data, row, col, shape = A.data, A.row, A.col, A.shape
        n = shape[0]
        assert n==shape[1]
        data2 = np.repeat(data, ncomp)
        nr = row.shape[0]
        row2 = np.repeat(ncomp*row, ncomp) + np.tile(np.arange(ncomp),nr).ravel()
        col2 = np.repeat(ncomp*col, ncomp) + np.tile(np.arange(ncomp),nr).ravel()
        return sparse.coo_matrix((data2, (row2, col2)), shape=(ncomp*n, ncomp*n)).tocsr()
    def matrix2system(self, A, ncomp, i, j):
        A = A.tocoo()
        data, row, col, shape = A.data, A.row, A.col, A.shape
        n = shape[0]
        assert n==shape[1]
        nr = row.shape[0]
        row2 = ncomp*row + i
        col2 = ncomp*col + j
        return sparse.coo_matrix((data, (row2, col2)), shape=(ncomp*n, ncomp*n)).tocsr()


# ------------------------------
    def test(self):
        import scipy.sparse.linalg as splinalg
        colors = self.mesh.bdrylabels.keys()
        bdrydata = self.prepareBoundary(colorsdirichlet=colors)
        A = self.computeMatrixElasticity(mucell=1, lamcell=10)
        A, bdrydata = self.matrixBoundaryStrong(A, bdrydata=bdrydata)
        b = np.zeros(self.nunknowns() * self.ncomp)
        rhs = lambda x, y, z: np.ones(shape=(self.ncomp, x.shape))
        # self.computeRhsCells(b, rhs)
        fp1 = self.interpolate(rhs)
        self.massDot(b, fp1, coeff=1)
        b = self.vectorBoundaryStrongZero(b, bdrydata)
        return splinalg.spsolve(A, b)


# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError(f"pas de test")
