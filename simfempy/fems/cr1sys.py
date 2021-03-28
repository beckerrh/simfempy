# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from simfempy.fems import femsys, cr1
from simfempy.tools import barycentric, npext

#=================================================================#
class CR1sys(femsys.Femsys):
    def __init__(self, ncomp, mesh=None):
        super().__init__(cr1.CR1(mesh), ncomp, mesh)
    def tonode(self, u):
        ncomp, nnodes = self.ncomp, self.mesh.nnodes
        unodes = np.zeros(ncomp*nnodes)
        for i in range(ncomp):
            unodes[i::ncomp] = self.fem.tonode(u[i::ncomp])
        return unodes
    def computeRhsCell(self, b, rhscell):
        ncomp = self.ncomp
        raise NotImplementedError()
    def computeRhsBoundary(self, b, bdryfct, colors):
        raise NotImplementedError()
        normals =  self.mesh.normals
        scale = 1
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            if not color in bdryfct or bdryfct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            b[faces] += scale * bdryfct[color](xf, yf, zf, nx, ny, nz) * dS
        return b
    def computeRhs(self, problemdata):
        ncomp = self.ncomp
        b = np.zeros(self.mesh.nfaces * self.ncomp)
        rhs = problemdata.params.fct_glob['rhs']
        if rhs:
            rhsall = self.fem.interpolate(rhs)
            for i in range(ncomp):
                self.fem.massDot(b[i::ncomp], rhsall[i])
        bdrycond = problemdata.bdrycond
        colorsneu = bdrycond.colorsOfType("Neumann")
        for color in colorsneu:
            if not color in bdrycond.fct or not bdrycond.fct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            scale = 1 / self.mesh.dimension
            dS = linalg.norm(normalsS, axis=1)
            xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
            x1, y1, z1 = xS[:, 0], xS[:, 1], xS[:, 2]
            nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
            neumanns = problemdata.bdrycond.fct[color](x1, y1, z1, nx, ny, nz)
            for i in range(ncomp):
                bS = scale * dS * neumanns[i]
                indices = i + self.ncomp * faces
                np.add.at(b, indices.T, bS)
        return b
    def computeMatrixDivergence(self):
        nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, cellgrads, facesOfCells = self.fem.nloc, self.fem.cellgrads, self.mesh.facesOfCells
        rowsB = np.repeat(np.arange(ncells), ncomp * nloc).reshape(ncells * nloc, ncomp)
        colsB = np.repeat(facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + nfaces * np.arange(ncomp)
        matB = (cellgrads[:, :, :ncomp].T * dV).T
        return sparse.coo_matrix((matB.reshape(-1), (rowsB.reshape(-1), colsB.reshape(-1))),
                                    shape=(ncells, nfaces * ncomp)).tocsr()
    def computeMatrixLaplace(self, mucell):
        nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, rows, cols, cellgrads = self.fem.nloc, self.rowssys, self.colssys, self.fem.cellgrads
        mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        for i in range(ncomp):
            mat[:, i::ncomp, i::ncomp] += (np.einsum('nkj,nlj->nkl', cellgrads, cellgrads).T * dV * mucell).T
        A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
        return A
    def computeMatrixElasticity(self, mucell, lamcell):
        nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, rows, cols, cellgrads = self.fem.nloc, self.rowssys, self.colssys, self.fem.cellgrads
        mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        for i in range(ncomp):
            for j in range(self.ncomp):
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, j]).T * dV * lamcell).T
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, i]).T * dV * mucell).T
                mat[:, i::ncomp, i::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, j]).T * dV * mucell).T
        A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
        A += self.computeMatrixKorn(mucell)
        return A
    def computeMatrixKorn(self, mucell):
        ncomp = self.ncomp
        dimension, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
        nloc, dofspercell, nall = self.nlocal(), self.dofspercell(), ncomp*ndofs
        ci0 = self.mesh.cellsOfInteriorFaces[:,0]
        ci1 = self.mesh.cellsOfInteriorFaces[:,1]
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)

        faces = self.mesh.faces[self.mesh.innerfaces]
        ind0 = npext.positionin(faces, self.mesh.simplices[ci0])
        ind1 = npext.positionin(faces, self.mesh.simplices[ci1])
        fi0 = np.take_along_axis(self.mesh.facesOfCells[ci0], ind0, axis=1)
        fi1 = np.take_along_axis(self.mesh.facesOfCells[ci1], ind1, axis=1)
        # fi0 = self.mesh.facesOfCells[ci0][ind0]
        # fi1 = self.mesh.facesOfCells[ci1][ind1]
        print(f"{self.mesh.facesOfCells[ci0].shape=}")
        print(f"{faces.shape=}")
        print(f"{self.mesh.simplices[ci0].shape=}")
        print(f"{ind0.shape=}")
        print(f"{fi0.shape=}")


        d = self.mesh.dimension
        massloc = barycentric.crbdryothers(d)
        if isinstance(mucell,float):
            scale = mucell*dS/(dV[ci0]+ dV[ci1])
        else:
            scale = (mucell[ci0] + mucell[ci1]) * dS / (dV[ci0] + dV[ci1])
        A = sparse.coo_matrix((nall, nall))
        mat = np.einsum('n,kl->nkl', dS*scale, massloc).reshape(-1)
        for icomp in range(ncomp):
            d0 = ncomp*fi0+icomp
            d1 = ncomp*fi1+icomp
            rows0 = d0.repeat(nloc-1)
            cols0 = np.tile(d0,nloc-1).reshape(-1)
            rows1 = d1.repeat(nloc-1)
            cols1 = np.tile(d1,nloc-1).reshape(-1)
            print(f"{mat.shape=}")
            print(f"{rows0.shape=}")
            print(f"{cols0.shape=}")
            print(f"{fi0.shape=}")
            print(f"{fi1.shape=}")
            A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(nall, nall))
            A += sparse.coo_matrix((-mat, (rows0, cols1)), shape=(nall, nall))
            A += sparse.coo_matrix((-mat, (rows1, cols0)), shape=(nall, nall))
            A += sparse.coo_matrix((mat, (rows1, cols1)), shape=(nall, nall))
        return A
    def vectorDirichlet(self, problemdata, method, b, u):
        bdrydata = self.bdrydata
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        for color, faces in facesdirflux.items():
            ind = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): ind[icomp::ncomp] += icomp
            bdrydata.bsaved[color] = b[ind]
        indin = np.repeat(ncomp * facesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        if method == 'trad':
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                if color in problemdata.bdrycond.fct.keys():
                    dirichlets = problemdata.bdrycond.fct[color](x[faces], y[faces], z[faces])
                    for icomp in range(ncomp):
                        b[icomp + ncomp * faces] = dirichlets[icomp]
                        u[icomp + ncomp * faces] = b[icomp + ncomp * faces]
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * faces] = 0
                        u[icomp + ncomp * faces] = b[icomp + ncomp * faces]
            b[indin] -= self.bdrydata.A_inner_dir * b[inddir]
        else:
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                if color in problemdata.bdrycond.fct.keys():
                    dirichlets = problemdata.bdrycond.fct[color](x[faces], y[faces], z[faces])
                    for icomp in range(ncomp):
                        u[icomp + ncomp * faces] = dirichlets[icomp]
                        b[icomp + ncomp * faces] = 0
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * faces] = 0
                        u[icomp + ncomp * faces] = b[icomp + ncomp * faces]
            b[indin] -= self.bdrydata.A_inner_dir * u[inddir]
            b[inddir] = self.bdrydata.A_dir_dir * u[inddir]
        # print(f"vectorDirichlet {self.bdrydata.bsaved.keys()=}")
        return b, u
    def matrixDirichlet(self, method, A):
        bdrydata = self.bdrydata
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        for color, faces in facesdirflux.items():
            ind = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): ind[icomp::ncomp] += icomp
            nb = faces.shape[0]
            help = sparse.dok_matrix((ncomp *nb, ncomp * nfaces))
            for icomp in range(ncomp):
                for i in range(nb): help[icomp + ncomp * i, icomp + ncomp * faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        indin = np.repeat(ncomp * facesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        bdrydata.A_inner_dir = A[indin, :][:, inddir]
        if method == 'trad':
            help = np.ones((ncomp * nfaces))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((ncomp * nfaces))
            help[inddir] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            A += help
        else:
            self.bdrydata.A_dir_dir = A[inddir, :][:, inddir]
            help = np.ones((ncomp * nfaces))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            help2 = np.zeros((ncomp * nfaces))
            help2[inddir] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A
    def computeBdryNormalFlux(self, u, colors):
        bdrydata = self.bdrydata
        flux, omega = np.zeros(shape=(len(colors),self.ncomp)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
            res = bs - As * u
            for icomp in range(self.ncomp):
                flux[i, icomp] = np.sum(res[icomp::self.ncomp])
        return flux

# ------------------------------------- #

if __name__ == '__main__':
    from simfempy.meshes import testmeshes
    from simfempy.meshes import plotmesh
    import matplotlib.pyplot as plt

    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = P1sys(mesh)
    u = fem.test()
    plotmesh.meshWithBoundaries(mesh)
    plotmesh.meshWithData(mesh, point_data=u, title="P1 Test", alpha=1)
    plt.show()
