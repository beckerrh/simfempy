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
        super().__init__(cr1.CR1(mesh=mesh), ncomp, mesh)
    def setMesh(self, mesh):
        super().setMesh(mesh)
    def tonode(self, u):
        ncomp, nnodes = self.ncomp, self.mesh.nnodes
        unodes = np.zeros(ncomp*nnodes)
        for i in range(ncomp):
            unodes[i::ncomp] = self.fem.tonode(u[i::ncomp])
        return unodes
    def interpolateBoundary(self, colors, f):
        # fs={col:f[col] for col in colors if col in f.keys()}
        if len(colors) == 0 or len(f) == 0: return
        # print(f"{f=}")
        if isinstance(next(iter(f.values())), list):
            import inspect
            fct = next(iter(f.values()))[0]
            # print(f"{str(inspect.signature(fct))=}")
            # print(f"{len(inspect.signature(fct).parameters)=}")
            # if 'nx' in str(inspect.signature(fct)):
            #     return np.vstack([self.fem.interpolateBoundary(colors, {col:np.vectorize(f[col][icomp], signature='(n),(n),(n),(n),(n),(n)->(n)') for col in colors if col in f.keys()},lumped) for icomp in range(self.ncomp)]).T
            # else:
            #     return np.vstack([self.fem.interpolateBoundary(colors, {col:np.vectorize(f[col][icomp]) for col in colors if col in f.keys()},lumped) for icomp in range(self.ncomp)]).T
            return np.vstack([self.fem.interpolateBoundary(colors, {col:f[col][icomp] for col in colors if col in f.keys()}) for icomp in range(self.ncomp)]).T
        else:
            raise ValueError(f"don't know how to handle {type(next(iter(f.values())))=}")
    def matrixBoundaryStrong(self, A, bdrydata, method):
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
        if method == 'strong':
            help = np.ones((ncomp * nfaces))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((ncomp * nfaces))
            help[inddir] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            A += help
        else:
            bdrydata.A_dir_dir = A[inddir, :][:, inddir]
            help = np.ones((ncomp * nfaces))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            help2 = np.zeros((ncomp * nfaces))
            help2[inddir] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nfaces, ncomp * nfaces))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A
    def formBoundary(self, b, bdrydata, method):
        facesdirall, ncomp = bdrydata.facesdirall, self.ncomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        b[inddir] = 0
    def vectorBoundaryStrong(self, b, bdryfct, bdrydata, method):
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
        help = np.zeros_like(b)
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            if color in bdryfct:
                # dirichlets = bdryfct[color](x[faces], y[faces], z[faces])
                dirichlets = np.vstack([f(x[faces], y[faces], z[faces]) for f in bdryfct[color]])
                for icomp in range(ncomp):
                    help[icomp + ncomp * faces] = dirichlets[icomp]
        b[indin] -= bdrydata.A_inner_dir * help[inddir]
        if method == 'strong':
            b[inddir] = help[inddir]
            # print(f"{b[inddir]=}")
        else:
            b[inddir] = bdrydata.A_dir_dir * help[inddir]
        return b
        # for color in colorsdir:
        #     faces = self.mesh.bdrylabels[color]
        #     if color in bdryfct.keys():
        #         dirichlets = bdryfct[color](x[faces], y[faces], z[faces])
        #         for icomp in range(ncomp):
        #             u[icomp + ncomp * faces] = dirichlets[icomp]
        #     else:
        #         for icomp in range(ncomp):
        #             u[icomp + ncomp * faces] = 0
        # b[indin] -= bdrydata.A_inner_dir * u[inddir]
        # if self.fem.dirichletmethod == 'strong':
        #     b[inddir] = u[inddir]
        # else:
        #     b[inddir] = bdrydata.A_dir_dir * u[inddir]
        # # print(f"{b=}")
        # return b, u, bdrydata
    def computeRhsBoundary(self, b, colors, bdryfct):
        for color in colors:
            if not color in bdryfct or not bdryfct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T / dS
            neumanns = bdryfct[color](xf, yf, zf, nx, ny, nz)
            for i in range(self.ncomp):
                bS = dS * neumanns[i]
                indices = i + self.ncomp * faces
                b[indices] += bS
        return b
    def computeMatrixDivergence(self):
        nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, cellgrads, foc = self.fem.nloc, self.fem.cellgrads, self.mesh.facesOfCells
        rowsB = np.repeat(np.arange(ncells), ncomp * nloc).ravel()
        colsB = ncomp*np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        mat = np.einsum('nkl,n->nkl', cellgrads[:, :, :ncomp], dV)
        B = sparse.coo_matrix((mat.ravel(), (rowsB, colsB.ravel())),shape=(ncells, nfaces * ncomp)).tocsr()
        return B
    def computeFormDivGrad(self, dv, dp, v, p):
        ncomp, dV, cellgrads, foc = self.ncomp, self.mesh.dV, self.fem.cellgrads, self.mesh.facesOfCells
        for icomp in range(ncomp):
            r = np.einsum('n,ni->ni', -dV*p, cellgrads[:,:,icomp])
            np.add.at(dv[icomp::ncomp], foc, r)
            dp += np.einsum('n,ni,ni->n', dV, cellgrads[:,:,icomp], v[icomp::ncomp][foc])
    def computeMatrixLaplace(self, mucell):
        return self.matrix2systemdiagonal(self.fem.computeMatrixDiffusion(mucell), self.ncomp).tocsr()
        # OLD VERSION: twice slower
        # import time
        # t0 = time.time()
        # nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        # nloc, rows, cols, cellgrads = self.fem.nloc, self.rowssys, self.colssys, self.fem.cellgrads
        # mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        # for icomp in range(ncomp):
        #     mat[:, icomp::ncomp, icomp::ncomp] += (np.einsum('nkj,nlj->nkl', cellgrads, cellgrads).T * dV * mucell).T
        # A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
        # t1 = time.time()

        # B = self.fem.computeMatrixDiffusion(mucell)
        # B = self.matrix2systemdiagonal(B, self.ncomp).tocsr()
        # t2 = time.time()
        # print(f"{(t2-t1)/(t1-t0)=}")
        # if not np.allclose(A.A,B.A):
        #     raise ValueError(f"{A.diagonal()=} {B.diagonal()=}")
        # return A
    def computeFormLaplace(self, mu, dv, v):
        ncomp, dV, cellgrads, foc = self.ncomp, self.mesh.dV, self.fem.cellgrads, self.mesh.facesOfCells
        for icomp in range(ncomp):
            r = np.einsum('n,nil,njl,nj->ni', dV*mu, cellgrads, cellgrads, v[icomp::ncomp][foc])
            np.add.at(dv[icomp::ncomp], foc, r)
    def computeRhsNitscheDiffusion(self, b, diffcoff, colors, udir, ncomp, nitsche_param):
        for icomp in range(ncomp):
            self.fem.computeRhsNitscheDiffusion(b[icomp::ncomp], diffcoff, colors, udir[:,icomp], nitsche_param)
    def computeRhsNitscheDiffusionNormal(self, b, diffcoff, colors, udir, ncomp, nitsche_param):
        faces = self.mesh.bdryFaces(colors)
        normalsS = self.mesh.normals[faces][:,:ncomp]
        dS = np.linalg.norm(normalsS, axis=1)
        normals = normalsS/dS[:,np.newaxis]
        if udir.shape[0] == self.mesh.nfaces*ncomp:
            for icomp in range(ncomp):
                for jcomp in range(ncomp):
                    self.fem.computeRhsNitscheDiffusion(b[icomp::ncomp], diffcoff, colors, udir[jcomp::ncomp], nitsche_param, coeff=normals[:,icomp]*normals[:,jcomp])
        else:
            assert udir.shape[0] == self.mesh.nfaces
            for icomp in range(ncomp):
                self.fem.computeRhsNitscheDiffusion(b[icomp::ncomp], diffcoff, colors, udir, nitsche_param, coeff=normals[:,icomp])
    def massDotBoundary(self, b, f, colors, ncomp, coeff=1):
        for icomp in range(ncomp):
            self.fem.massDotBoundary(b[icomp::ncomp], f[icomp::ncomp], colors=colors, coeff=coeff)
    def massDotBoundaryNormal(self, b, f, colors, ncomp, coeff=1):
        faces = self.mesh.bdryFaces(colors)
        normalsS = self.mesh.normals[faces][:,:ncomp]
        dS = np.linalg.norm(normalsS, axis=1)
        normals = normalsS/dS[:,np.newaxis]
        if f.shape[0] == self.mesh.nfaces*ncomp:
            for icomp in range(ncomp):
                for jcomp in range(ncomp):
                    self.fem.massDotBoundary(b[icomp::ncomp], f[jcomp::ncomp], colors=colors, coeff=normals[:,icomp]*normals[:,jcomp]*coeff)
        else:
            assert f.shape[0] == self.mesh.nfaces
            for icomp in range(ncomp):
                self.fem.massDotBoundary(b[icomp::ncomp], f, colors=colors, coeff=normals[:,icomp]*coeff)

    def computeMassMatrixBoundary(self, colors, ncomp, coeff):
        A = self.fem.computeBdryMassMatrix(colors, coeff)
        return self.matrix2systemdiagonal(A, ncomp)
    def computeMassMatrixBoundaryNormal(self, colors, ncomp, coeff):
        nfaces, ncells, dim  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension
        assert dim == ncomp
        faces = self.mesh.bdryFaces(colors)
        normalsS = self.mesh.normals[faces][:, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        A = sparse.coo_matrix((ncomp*nfaces, ncomp*nfaces))
        for i in range(ncomp):
            for j in range(i,ncomp):
                Aij = self.fem.computeBdryMassMatrix(colors, coeff*normalsS[:,i]*normalsS[:,j]/dS**2)
                A += self.matrix2system(Aij, ncomp, i, j)
                if i!=j: A += self.matrix2system(Aij, ncomp, j, i)
        return A
    def computeMatrixNitscheDiffusion(self, diffcoff, colors, ncomp, nitsche_param):
        A = self.fem.computeMatrixNitscheDiffusion(diffcoff, colors, nitsche_param)
        return self.matrix2systemdiagonal(A, ncomp)
    def computeMatrixNitscheDiffusionNormal(self, diffcoff, colors, ncomp, nitsche_param):
        nfaces, ncells, dim  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension
        assert dim == ncomp
        faces = self.mesh.bdryFaces(colors)
        # cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        A = sparse.coo_matrix((ncomp*nfaces, ncomp*nfaces))
        for i in range(ncomp):
            for j in range(i,ncomp):
                Aij = self.fem.computeMatrixNitscheDiffusion(diffcoff, colors, nitsche_param=nitsche_param, coeff=normalsS[:,i]*normalsS[:,j]/dS**2)
                A += self.matrix2system(Aij, ncomp, i, j)
                if i!=j: A += self.matrix2system(Aij, ncomp, j, i)
        return A
    def computeFormNitscheDiffusion(self, du, u, diffcoff, colorsdir, ncomp, nitsche_param):
        for icomp in range(ncomp):
            self.fem.computeFormNitscheDiffusion(du[icomp::ncomp], u[icomp::ncomp], diffcoff, colorsdir, nitsche_param)


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
        assert np.all(ci1>=0)
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        faces = self.mesh.faces[self.mesh.innerfaces]
        ind0 = npext.positionin(faces, self.mesh.simplices[ci0])
        ind1 = npext.positionin(faces, self.mesh.simplices[ci1])
        fi0 = np.take_along_axis(self.mesh.facesOfCells[ci0], ind0, axis=1)
        fi1 = np.take_along_axis(self.mesh.facesOfCells[ci1], ind1, axis=1)
        d = self.mesh.dimension
        massloc = barycentric.crbdryothers(d)
        if isinstance(mucell,(int,float)):
            scale = mucell*dS/(dV[ci0]+ dV[ci1])
        else:
            scale = (mucell[ci0] + mucell[ci1]) * dS / (dV[ci0] + dV[ci1])
        scale *= 8
        A = sparse.coo_matrix((nall, nall))
        mat = np.einsum('n,kl->nkl', dS*scale, massloc).reshape(-1)
        for icomp in range(ncomp):
            d0 = ncomp*fi0+icomp
            d1 = ncomp*fi1+icomp
            rows0 = d0.repeat(nloc-1)
            cols0 = np.tile(d0,nloc-1).reshape(-1)
            rows1 = d1.repeat(nloc-1)
            cols1 = np.tile(d1,nloc-1).reshape(-1)
            # print(f"{mat.shape=}")
            # print(f"{rows0.shape=}")
            # print(f"{cols0.shape=}")
            # print(f"{fi0.shape=}")
            # print(f"{fi1.shape=}")
            A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(nall, nall))
            A += sparse.coo_matrix((-mat, (rows0, cols1)), shape=(nall, nall))
            A += sparse.coo_matrix((-mat, (rows1, cols0)), shape=(nall, nall))
            A += sparse.coo_matrix((mat, (rows1, cols1)), shape=(nall, nall))
        return A
    def computeBdryNormalFlux(self, u, colors, bdrydata):
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
    import matplotlib.gridspec as gridspec

    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = CR1sys(ncomp=2, mesh=mesh)
    u = fem.test()
    point_data = fem.getPointData(u)
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    plotmesh.meshWithBoundaries(mesh, fig=fig, outer=outer[0])
    plotmesh.meshWithData(mesh, point_data=point_data, title="P1 Test", alpha=1, fig=fig, outer=outer[1])
    plt.show()
