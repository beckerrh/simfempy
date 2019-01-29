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
    def setMesh(self, mesh, ncomp):
        self.mesh = mesh
        self.ncomp = ncomp
        self.nloc = self.mesh.dimension+1
        ncells, facesOfCells = self.mesh.ncells, self.mesh.facesOfCells
        nlocncomp = ncomp * self.nloc
        self.rows = np.repeat(ncomp * facesOfCells, ncomp).reshape(ncells * self.nloc, ncomp) + np.arange(ncomp)
        self.rows = self.rows.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.cols = self.rows.swapaxes(1, 2)
        # self.cols = self.cols.flatten()
        # self.rows = self.rows.flatten()
        self.cols = self.cols.reshape(-1)
        self.rows = self.rows.reshape(-1)
        self.computeFemMatrices()
        self.massmatrix = self.massMatrix()
        self.pointsf = self.mesh.points[self.mesh.faces].mean(axis=1)

    def computeRhs(self, rhs, solexact, diff, bdrycond):
        b = np.zeros(self.mesh.nfaces * self.ncomp)
        if solexact or rhs:
            x, y, z = self.pointsf[:,0], self.pointsf[:,1], self.pointsf[:,2]
            for icomp in range(self.ncomp):
                if solexact:
                    bfaces = -solexact[icomp].xx(x, y, z) - solexact[icomp].yy(x, y, z) - solexact[icomp].zz(x, y, z)
                    bfaces *= diff[icomp][0]
                else:
                    bfaces = rhs(x, y, z)
                b[icomp::self.ncomp] = self.massmatrix * bfaces
        normals =  self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            for icomp in range(self.ncomp):
                condition = bdrycond[icomp].type[color]
                if condition == "Neumann":
                    neumann = bdrycond[icomp].fct[color]
                    normalsS = normals[faces]
                    dS = linalg.norm(normalsS,axis=1)
                    kS = diff[icomp][self.mesh.cellsOfFaces[faces,0]]
                    x1, y1, z1 = self.pointsf[faces,0], self.pointsf[faces,1], self.pointsf[faces,2]
                    nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                    if solexact:
                        bS = dS*kS*(solexact[icomp].x(x1, y1, z1)*nx + solexact[icomp].y(x1, y1, z1)*ny + solexact[icomp].z(x1, y1, z1)*nz)
                    else:
                        bS = neumann(x1, y1, z1, nx, ny, nz, kS) * dS
                    b[icomp+self.ncomp*faces] += bS
        return b

    def massMatrix(self):
        nfaces = self.mesh.nfaces
        cols = np.tile(self.mesh.facesOfCells, self.nloc).reshape(self.mesh.ncells, self.nloc, self.nloc)
        rows = cols.swapaxes(1, 2)
        self.massmatrix = sparse.coo_matrix((self.mass, (rows.flatten(), cols.flatten())), shape=(nfaces, nfaces)).tocsr()
        return self.massmatrix

    def matrixDiffusion(self, k):
        nfaces, ncells, ncomp = self.mesh.nfaces, self.mesh.ncells, self.ncomp
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = np.zeros(shape=self.rows.shape, dtype=float).reshape(ncells, ncomp * self.nloc, ncomp * self.nloc)
        for icomp in range(ncomp):
            mat[:, icomp::ncomp, icomp::ncomp] = ((matxx + matyy + matzz).T * self.mesh.dV * k[icomp]).T
        return sparse.coo_matrix((mat.flatten(), (self.rows, self.cols)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()

    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = 1
        self.cellgrads = scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
        dim = self.mesh.dimension
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        self.mass = np.einsum('n,kl->nkl', dV, massloc).flatten()

    def prepareBoundary(self, colorsdir, postproc):
        facesdirall = np.empty(shape=(0), dtype=int)
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            facesdirall = np.unique(np.union1d(facesdirall, facesdir))
        facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int),facesdirall)
        facesdirflux={}
        for key, val in postproc.items():
            type,data = val.split(":")
            if type != "bdrydn": continue
            colors = [int(x) for x in data.split(',')]
            facesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                facesdir = self.mesh.bdrylabels[color]
                facesdirflux[key] = np.unique(np.union1d(facesdirflux[key], facesdir).flatten())
        return facesdirall, facesinner, colorsdir, facesdirflux

    def boundary(self, A, b, u, bdrycond, bdrydata, method):
        x, y, z = self.pointsf[:, 0], self.pointsf[:, 1], self.pointsf[:, 2]
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        self.bsaved = []
        self.Asaved = []
        for icomp in range(ncomp):
            facesdirall, facesinner, colorsdir, facesdirflux = bdrydata[icomp]
            self.bsaved.append({})
            self.Asaved.append({})
            for key, faces in facesdirflux.items():
                self.bsaved[icomp][key] = b[icomp + ncomp * faces]
            for key, faces in facesdirflux.items():
                nb = faces.shape[0]
                help = sparse.dok_matrix((nb, ncomp * nfaces))
                for i in range(nb): help[i, icomp + ncomp * faces[i]] = 1
                self.Asaved[icomp][key] = help.dot(A)
            if method == 'trad':
                for color in colorsdir:
                    faces = self.mesh.bdrylabels[color]
                    dirichlet = bdrycond[icomp].fct[color]
                    b[icomp + ncomp * faces] = dirichlet(x[faces], y[faces], z[faces])
                    u[icomp + ncomp * faces] = b[icomp + ncomp * faces]
                indin = icomp + ncomp *facesinner
                inddir = icomp + ncomp *facesdirall
                b[indin] -= A[indin, :][:,inddir] * b[inddir]
                help = np.ones((ncomp * nfaces))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A = help.dot(A.dot(help))
                help = np.zeros((ncomp * nfaces))
                help[inddir] = 1.0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A += help
            else:
                for color in colorsdir:
                    faces = self.mesh.bdrylabels[color]
                    dirichlet = bdrycond[icomp].fct[color]
                    u[icomp + ncomp * faces] = dirichlet(x[faces], y[faces], z[faces])
                    b[icomp + ncomp * faces] = 0
                indin = icomp + ncomp *facesinner
                inddir = icomp + ncomp *facesdirall
                b[indin] -= A[indin, :][:, inddir] * u[inddir]
                b[inddir] = A[inddir, :][:, inddir] * u[inddir]
                help = np.ones((ncomp * nfaces))
                help[inddir] = 0
                # print("help", help)
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                help2 = np.zeros((ncomp * nfaces))
                help2[inddir] = 1
                # print("help2", help2)
                help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nfaces, ncomp * nfaces))
                A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, b, u

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
        e = []
        err = []
        for icomp in range(self.ncomp):
            e.append(solex[icomp](x, y, z) - uh[icomp::self.ncomp])
            err.append(np.sqrt(np.dot(e[icomp], self.massmatrix * e[icomp])))
        return err, e

    def computeBdryMean(self, u, key, data, icomp):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            mean += np.sum(dS * u[icomp + self.ncomp * faces])
        return mean

    def computeBdryDn(self, u, key, data, icomp):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[icomp][key] - self.Asaved[icomp][key] * u)
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


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemCR1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
