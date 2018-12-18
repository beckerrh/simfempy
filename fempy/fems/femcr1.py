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
class FemCR1(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        self.cols = np.tile(self.mesh.facesOfCells, self.nloc).flatten()
        self.rows = np.repeat(self.mesh.facesOfCells, self.nloc).flatten()
        self.computeFemMatrices()
        self.massmatrix = self.massMatrix()
        self.pointsf = self.mesh.points[self.mesh.faces].mean(axis=1)
    def prepareBoundary(self, colorsdir, postproc):
        self.facesdirall = np.empty(shape=(0), dtype=int)
        self.colorsdir = colorsdir
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            self.facesdirall = np.unique(np.union1d(self.facesdirall, facesdir))
        self.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int),self.facesdirall)
        self.bsaved={}
        self.Asaved={}
        self.facesdirflux={}
        for key, val in postproc.items():
            type,data = val.split(":")
            if type != "flux": continue
            colors = [int(x) for x in data.split(',')]
            self.facesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                facesdir = self.mesh.bdrylabels[color]
                self.facesdirflux[key] = np.unique(np.union1d(self.facesdirflux[key], facesdir).flatten())
    def computeRhs(self, rhs, solexact, kheatcell, bdrycond):
        x, y, z = self.pointsf[:,0], self.pointsf[:,1], self.pointsf[:,2]
        if solexact:
            bnodes = -solexact.xx(x, y, z) - solexact.yy(x, y, z)- solexact.zz(x, y, z)
            bnodes *= kheatcell[0]
        else:
            bnodes = rhs(x, y, z)
        b = self.massmatrix*bnodes
        normals =  self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = bdrycond.type[color]
            if condition == "Neumann":
                neumann = bdrycond.fct[color]
                normalsS = normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                kS = kheatcell[self.mesh.cellsOfFaces[faces,0]]
                assert(dS.shape[0] == len(faces))
                assert(kS.shape[0] == len(faces))
                x1, y1, z1 = self.pointsf[faces,0], self.pointsf[faces,1], self.pointsf[faces,2]
                nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                if solexact:
                    bS = dS*kS*(solexact.x(x1, y1, z1)*nx + solexact.y(x1, y1, z1)*ny + solexact.z(x1, y1, z1)*nz)
                else:
                    bS = neumann(x1, y1, z1, nx, ny, nz, kS) * dS
                b[faces] += bS
                # np.add.at(b, faces.T, bS)
        return b
    def massMatrix(self):
        nfaces = self.mesh.nfaces
        self.massmatrix = sparse.coo_matrix((self.mass, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
        return self.massmatrix
    def matrixDiffusion(self, k):
        nfaces = self.mesh.nfaces
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = ( (matxx+matyy+matzz).T*self.mesh.dV*k).T.flatten()
        return sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = -1
        self.cellgrads = scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
        dim = self.mesh.dimension
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        self.mass = np.einsum('n,kl->nkl', dV, massloc).flatten()
    def boundary(self, A, b, bdrycond):
        x, y, z = self.pointsf[:, 0], self.pointsf[:, 1], self.pointsf[:, 2]
        nfaces = self.mesh.nfaces
        for key, faces in self.facesdirflux.items():
            self.bsaved[key] = b[faces]
        for color in self.colorsdir:
            faces = self.mesh.bdrylabels[color]
            dirichlet = bdrycond.fct[color]
            b[faces] = dirichlet(x[faces], y[faces], z[faces])
        for key, faces in self.facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            self.Asaved[key] = help.dot(A)
        help = np.ones((nfaces))
        help[self.facesdirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
        b[self.facesinner] -= A[self.facesinner, :][:, self.facesdirall] * b[self.facesdirall]
        A = help.dot(A.dot(help))
        help = np.zeros((nfaces))
        help[self.facesdirall] = 1.0
        help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
        A += help
        return A, b
    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.facesOfCells[ic,:]]
        grads = -normals/self.mesh.dV[ic]
        chsg =  (ic == self.mesh.cellsOfFaces[self.mesh.facesOfCells[ic,:],0])
        # print("### chsg", chsg, "normals", normals)
        grads[chsg] *= -1.
        return grads
    def phi(self, ic, x, y, z, grad):
        return 1./3. + np.dot(grad, np.array([x-self.mesh.pointsc[ic,0], y-self.mesh.pointsc[ic,1], z-self.mesh.pointsc[ic,2]]))
    def testgrad(self):
        for ic in range(self.mesh.ncells):
            grads = self.grad(ic)
            for ii in range(3):
                x = self.pointsf[self.mesh.facesOfCells[ic,ii], 0]
                y = self.pointsf[self.mesh.facesOfCells[ic,ii], 1]
                z = self.pointsf[self.mesh.facesOfCells[ic,ii], 2]
                for jj in range(3):
                    phi = self.phi(ic, x, y, z, grads[jj])
                    if ii == jj:
                        test = np.abs(phi-1.0)
                        if test > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            print('x,y', x, y)
                            print('x-xc,y-yc', x-self.mesh.pointsc[ic,0], y-self.mesh.pointsc[ic,1])
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic,ii,jj, test))
                    else:
                        test = np.abs(phi)
                        if np.abs(phi) > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic,ii,jj, test))
    def computeErrorL2(self, solex, uh):
        x, y, z = self.pointsf[:,0], self.pointsf[:,1], self.pointsf[:,2]
        e = solex(x, y, z) - uh
        return np.sqrt( np.dot(e, self.massmatrix*e) ), e
    def computeMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            mean += np.sum(dS*u[faces])
        return mean
    def computeFlux(self, u, key, data):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[key] - self.Asaved[key]*u )
        return flux
    def tonode(self, u):
        unodes = np.zeros(self.mesh.nnodes)
        np.add.at(unodes, self.mesh.faces.T, u)
        return unodes


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemCR1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
