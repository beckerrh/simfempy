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

    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        self.cols = np.tile(self.mesh.facesOfCells, self.nloc).flatten()
        self.rows = np.repeat(self.mesh.facesOfCells, self.nloc).flatten()
        self.computeFemMatrices()
        self.massmatrix = self.massMatrix()

    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        dim = self.mesh.dimension
        scale = 1
        self.cellgrads = scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        self.mass = np.einsum('n,kl->nkl', dV, massloc).flatten()

    def massMatrix(self):
        nfaces = self.mesh.nfaces
        self.massmatrix = sparse.coo_matrix((self.mass, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        return self.massmatrix

    def computeRhs(self, u, rhs, kheatcell, bdrycond, method, bdrydata):
        b = np.zeros(self.mesh.nfaces)
        if rhs:
            x, y, z = self.mesh.pointsf.T
            bnodes = rhs(x, y, z, kheatcell[0])
            b += self.massmatrix * bnodes
        normals =  self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] not in ["Neumann","Robin"]: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            kS = kheatcell[self.mesh.cellsOfFaces[faces,0]]
            assert(dS.shape[0] == len(faces))
            assert(kS.shape[0] == len(faces))
            # xf, yf, zf = self.pointsf[faces,0], self.pointsf[faces,1], self.pointsf[faces,2]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
            bS = bdrycond.fct[color](xf, yf, zf, nx, ny, nz, kS) * dS
            b[faces] += bS
        return self.vectorDirichlet(b, u, bdrycond, method, bdrydata)

    def matrixDiffusion(self, k, bdrycond, method, bdrydata):
        nfaces = self.mesh.nfaces
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = ( (matxx+matyy+matzz).T*self.mesh.dV*k).T.flatten()
        rows = np.copy(self.rows)
        cols = np.copy(self.cols)
        mat, rows, cols = self.matrixRobin(mat, rows, cols, bdrycond, method, bdrydata)
        A = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
        # A = sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        return self.matrixDirichlet(A, bdrycond, method, bdrydata)

    def matrixRobin(self, mat, rows, cols, bdrycond, method, bdrydata):
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] != "Robin": continue
            scalemass = bdrycond.param[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            cols = np.append(cols, faces)
            rows = np.append(rows, faces)
            mat = np.append(mat, scalemass*dS)
        return mat, rows, cols

    def prepareBoundary(self, colorsdir, postproc):
        bdrydata = simfempy.fems.bdrydata.BdryData()
        bdrydata.facesdirall = np.empty(shape=(0), dtype=int)
        bdrydata.colorsdir = colorsdir
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.facesdirall = np.unique(np.union1d(bdrydata.facesdirall, facesdir))
        bdrydata.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int), bdrydata.facesdirall)
        bdrydata.facesdirflux = {}
        for key, val in postproc.items():
            type, data = val.split(":")
            if type != "bdrydn": continue
            colors = [int(x) for x in data.split(',')]
            bdrydata.facesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                facesdir = self.mesh.bdrylabels[color]
                bdrydata.facesdirflux[key] = np.unique(np.union1d(bdrydata.facesdirflux[key], facesdir).flatten())
        return bdrydata

    def vectorDirichlet(self, b, u, bdrycond, method, bdrydata):
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        nfaces = self.mesh.nfaces
        for key, faces in facesdirflux.items():
            bdrydata.bsaved[key] = b[faces]
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
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            bdrydata.Asaved[key] = help.dot(A)
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

    def computeErrorL2(self, solex, uh):
        x, y, z = self.mesh.pointsf.T
        e = solex(x, y, z) - uh
        return np.sqrt( np.dot(e, self.massmatrix*e) ), e

    def computeBdryMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega += np.sum(dS)
            mean += np.sum(dS*u[faces])
        return mean/omega

    def computeBdryDn(self, u, key, data, bs, As):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(bs - As*u )
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
