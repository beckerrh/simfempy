# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
try:
    from simfempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh
import simfempy.fems.bdrydata


#=================================================================#
class FemCR1(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)
        self.dirichlet_al = 10

    def setMesh(self, mesh, bdrycond=None):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        self.cols = np.tile(self.mesh.facesOfCells, self.nloc).ravel()
        self.rows = np.repeat(self.mesh.facesOfCells, self.nloc).ravel()
        self.computeFemMatrices()
        self.massmatrix = self.massMatrix()
        if bdrycond:
            self.robinmassmatrix = self.computeBdryMassMatrix(bdrycond, type="Robin")

    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        dim = self.mesh.dimension
        scale = 1
        self.cellgrads = scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        self.mass = np.einsum('n,kl->nkl', dV, massloc).ravel()

    def massMatrix(self):
        nfaces = self.mesh.nfaces
        self.massmatrix = sparse.coo_matrix((self.mass, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        return self.massmatrix

    # def computeRhs(self, u, rhs, kheatcell, bdrycond, method, bdrydata):
    def computeRhs(self, u, problemdata, kheatcell, method, bdrydata, robinmassmatrix):
        rhs = problemdata.rhs
        rhscell = problemdata.rhscell
        rhspoint = problemdata.rhspoint
        bdrycond = problemdata.bdrycond
        normals =  self.mesh.normals
        b = np.zeros(self.mesh.nfaces)
        if rhscell:
            x, y, z = self.mesh.pointsf.T
            b += self.massmatrix * rhscell(x, y, z)
        help = np.zeros(self.mesh.nfaces)
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] != "Robin": continue
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = np.mean(normals[faces], axis=0)
            help[faces] = bdrycond.fct[color](xf, yf, zf, nx, ny, nz)
        b += robinmassmatrix*help
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] != "Neumann": continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            b[faces] += bdrycond.fct[color](xf, yf, zf, nx, ny, nz) * dS
        if bdrycond.hasExactSolution():
            for color, faces in self.mesh.bdrylabels.items():
                if bdrycond.type[color] != "Robin": continue
                normalsS = normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                normalsS = normalsS/dS[:,np.newaxis]
                xf, yf, zf = self.mesh.pointsf[faces].T
                nx, ny, nz = normalsS.T
                b[faces] += bdrycond.fctexact["Robin"](xf, yf, zf, nx, ny, nz) * dS
        return self.vectorDirichlet(b, u, bdrycond, method, bdrydata)
    def matrixDiffusion(self, k, bdrycond, method, bdrydata):
        nfaces = self.mesh.nfaces
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = ( (matxx+matyy+matzz).T*self.mesh.dV*k).T.ravel()
        A = sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        return A
    def computeBdryMassMatrix(self, bdrycond, bdrycondtype, lumped):
        nfaces = self.mesh.nfaces
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] != type: continue
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            cols = np.append(cols, faces)
            rows = np.append(rows, faces)
            mat = np.append(mat, bdrycond.param[color] * dS)
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
    def prepareBoundary(self, colorsdir, postproc):
        bdrydata = simfempy.fems.bdrydata.BdryData()
        bdrydata.facesdirall = np.empty(shape=(0), dtype=int)
        bdrydata.colorsdir = colorsdir
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.facesdirall = np.unique(np.union1d(bdrydata.facesdirall, facesdir))
        bdrydata.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int), bdrydata.facesdirall)
        bdrydata.facesdirflux = {}
        # for key, val in postproc.items():
        #     type, data = val.split(":")
        #     if type != "bdrydn": continue
        #     colors = [int(x) for x in data.split(',')]
        #     # bdrydata.facesdirflux[key] = np.empty(shape=(0), dtype=int)
        #     for color in colors:
        #         facesdir = self.mesh.bdrylabels[color]
        #         # bdrydata.facesdirflux[key] = np.unique(np.union1d(bdrydata.facesdirflux[key], facesdir).ravel())
        #         bdrydata.facesdirflux[color] = facesdir
        for name, type in postproc.type.items():
            if type != "bdrydn": continue
            colors = postproc.colors(name)
            for color in colors:
                facesdir = self.mesh.bdrylabels[color]
                bdrydata.facesdirflux[color] = facesdir
        return bdrydata
    def vectorDirichlet(self, b, u, bdrycond, method, bdrydata):
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        nfaces = self.mesh.nfaces
        # for key, faces in facesdirflux.items():
        #     bdrydata.bsaved[key] = b[faces]
        for color, faces in facesdirflux.items():
            bdrydata.bsaved[color] = b[faces]
        if method == 'trad':
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                dirichlet = bdrycond.fct[color]
                b[faces] = dirichlet(x[faces], y[faces], z[faces])
                u[faces] = b[faces]
            b[facesinner] -= bdrydata.A_inner_dir * b[facesdirall]
        else:
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                dirichlet = bdrycond.fct[color]
                u[faces] = dirichlet(x[faces], y[faces], z[faces])
                # b[faces] = 0
            b[facesinner] -= bdrydata.A_inner_dir * u[facesdirall]
            b[facesdirall] += bdrydata.A_dir_dir * u[facesdirall]
        return b, u, bdrydata
    def matrixDirichlet(self, A, bdrycond, method, bdrydata):
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        nfaces = self.mesh.nfaces
        # for key, faces in facesdirflux.items():
        #     nb = faces.shape[0]
        #     help = sparse.dok_matrix((nb, nfaces))
        #     for i in range(nb): help[i, faces[i]] = 1
        #     bdrydata.Asaved[key] = help.dot(A)
        for color, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        if method == 'trad':
            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((nfaces))
            help[facesdirall] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A += help
        else:
            bdrydata.A_dir_dir = self.dirichlet_al*A[facesdirall, :][:, facesdirall]
            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            help2 = np.zeros((nfaces))
            help2[facesdirall] = np.sqrt(self.dirichlet_al)
            help2 = sparse.dia_matrix((help2, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, bdrydata
    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.facesOfCells[ic,:]]
        grads = -normals/self.mesh.dV[ic]
        chsg =  (ic == self.mesh.cellsOfFaces[self.mesh.facesOfCells[ic,:],0])
        # print("### chsg", chsg, "normals", normals)
        grads[chsg] *= -1.
        return grads
    def computeErrorL2(self, solexact, uh):
        x, y, z = self.mesh.pointsf.T
        xc, yc, zc = self.mesh.pointsc.T
        e = solexact(x, y, z) - uh
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.facesOfCells], axis=1)
        return np.sqrt( np.dot(e, self.massmatrix*e) ), np.sqrt(np.sum(ec**2* self.mesh.dV)), e
    def computeErrorFluxL2(self, solexact, diffcell, uh):
        xc, yc, zc = self.mesh.pointsc.T
        graduh = np.einsum('nij,ni->nj', self.cellgrads, uh[self.mesh.facesOfCells])
        errv = 0
        for i in range(self.mesh.dimension):
            solxi = solexact.d(i, xc, yc, zc)
            errv += np.sum( diffcell*(solxi-graduh[:,i])**2* self.mesh.dV)
        return np.sqrt(errv)
    def computeBdryMean(self, u, colors):
        # colors = [int(x) for x in data.split(',')]
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*u[faces])
        return mean/omega
    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * u[faces])
        xf, yf, zf = self.mesh.pointsf[faces].T
        normalsS = self.mesh.normals[faces]
        dS = linalg.norm(normalsS, axis=1)
        normalsS = normalsS / dS[:, np.newaxis]
        assert (dS.shape[0] == len(faces))
        nx, ny, nz = normalsS.T
        if uR: uRmean =  np.sum(dS * uR(xf, yf, zf, nx, ny, nz))
        else: uRmean=0
        return cR*(uRmean-uhmean)

    def computeBdryDn(self, u, colors, bdrydata, bdrycond):
        # colors = [int(x) for x in data.split(',')]
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if bdrycond.type[color] == "Robin":
                flux[i] = self.comuteFluxOnRobin(u, faces, dS, bdrycond.fct[color], bdrycond.param[color])
            elif bdrycond.type[color] == "Dirichlet":
                bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
                flux[i] = np.sum(As * u - bs)
            else:
                raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
        return flux

    def tonode(self, u):
        unodes = np.zeros(self.mesh.nnodes)
        scale = self.mesh.dimension
        np.add.at(unodes, self.mesh.simplices.T, np.sum(u[self.mesh.facesOfCells], axis=1))
        np.add.at(unodes, self.mesh.simplices.T, -scale*u[self.mesh.facesOfCells].T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.simplices.T, 1)
        unodes /= countnodes
        return unodes


#=================================================================#
if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemCR1(trimesh)
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
