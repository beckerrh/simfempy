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


# =================================================================#
class FemP1(object):
    def __init__(self, mesh=None):
        if mesh is not None:
            self.setMesh(mesh)

    def setMesh(self, mesh, ncomp):
        self.mesh = mesh
        self.ncomp = ncomp
        self.nloc = self.mesh.dimension + 1
        ncells, simps = self.mesh.ncells, self.mesh.simplices
        # plus rapide pour ncells petit mais plus lent pour ncells grand (?!)
        # self.cols = np.tile(simps, self.nloc).flatten()
        # self.rows2 = np.repeat(simps, self.nloc).flatten()
        #
        # self.cols = np.tile(simps, self.nloc).reshape(self.mesh.ncells, self.nloc*ncomp, self.nloc*ncomp)
        # self.rows = self.cols.swapaxes(1,2)
        # self.cols = self.cols.flatten()
        # self.rows = self.rows.flatten()

        # print("simps", simps)
        nlocncomp = ncomp * self.nloc
        self.rows = np.repeat(ncomp * simps, ncomp).reshape(ncells * self.nloc, ncomp) + np.arange(ncomp)
        self.rows = self.rows.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.cols = self.rows.swapaxes(1, 2)
        self.cols = self.cols.flatten()
        self.rows = self.rows.flatten()
        self.computeFemMatrices()
        self.massmatrix = self.massMatrix()

    def prepareBoundary(self, colorsdir, postproc):
        nodesdir = {}
        nodedirall = np.empty(shape=(0), dtype=int)
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            nodesdir[color] = np.unique(self.mesh.faces[facesdir].flat[:])
            nodedirall = np.unique(np.union1d(nodedirall, nodesdir[color]))
        nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=int), nodedirall)
        nodesdirflux = {}
        for key, val in postproc.items():
            type, data = val.split(":")
            if type != "bdrydn": continue
            colors = [int(x) for x in data.split(',')]
            nodesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                facesdir = self.mesh.bdrylabels[color]
                nodesdirflux[key] = np.unique(
                    np.union1d(nodesdirflux[key], np.unique(self.mesh.faces[facesdir].flatten())))
        return nodedirall, nodesinner, nodesdir, nodesdirflux

    def computeRhs(self, rhs, solexact, diff, bdrycond):
        b = np.zeros(self.mesh.nnodes * self.ncomp)
        if solexact or rhs:
            x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
            for icomp in range(self.ncomp):
                if solexact:
                    bnodes = -solexact[icomp].xx(x, y, z) - solexact[icomp].yy(x, y, z) - solexact[icomp].zz(x, y, z)
                    bnodes *= diff[icomp][0]
                else:
                    bnodes = rhs[icomp](x, y, z)
                b[icomp::self.ncomp] = self.massmatrix * bnodes
        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            for icomp in range(self.ncomp):
                condition = bdrycond[icomp].type[color]
                if condition == "Neumann":
                    neumann = bdrycond[icomp].fct[color]
                    scale = 1 / self.mesh.dimension
                    normalsS = normals[faces]
                    dS = linalg.norm(normalsS, axis=1)
                    xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
                    kS = diff[icomp][self.mesh.cellsOfFaces[faces, 0]]
                    x1, y1, z1 = xS[:, 0], xS[:, 1], xS[:, 2]
                    nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
                    if solexact:
                        bS = scale * dS * kS * (
                                    solexact[icomp].x(x1, y1, z1) * nx + solexact[icomp].y(x1, y1, z1) * ny + solexact[
                                icomp].z(x1, y1, z1) * nz)
                    else:
                        bS = scale * neumann(x1, y1, z1, nx, ny, nz, kS) * dS
                    # print("self.mesh.faces[faces]",self.mesh.faces[faces])
                    indices = icomp + self.ncomp * self.mesh.faces[faces]
                    # print("indices",indices)
                    # print("b",b)
                    # print("bS",bS)
                    np.add.at(b, indices.T, bS)
        return b

    def massMatrix(self):
        nnodes = self.mesh.nnodes
        cols = np.tile(self.mesh.simplices, self.nloc).reshape(self.mesh.ncells, self.nloc, self.nloc)
        rows = cols.swapaxes(1, 2)
        self.massmatrix = sparse.coo_matrix((self.mass, (rows.flatten(), cols.flatten())),
                                            shape=(nnodes, nnodes)).tocsr()
        return self.massmatrix

    def matrixDiffusion(self, k):
        nnodes, ncells, ncomp = self.mesh.nnodes, self.mesh.ncells, self.ncomp
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = np.zeros(shape=self.rows.shape, dtype=float).reshape(ncells, ncomp * self.nloc, ncomp * self.nloc)
        for icomp in range(ncomp):
            mat[:, icomp::ncomp, icomp::ncomp] = ((matxx + matyy + matzz).T * self.mesh.dV * k[icomp]).T
        return sparse.coo_matrix((mat.flatten(), (self.rows, self.cols)), shape=(ncomp*nnodes, ncomp*nnodes)).tocsr()

    def computeFemMatrices(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = 1 / self.mesh.dimension
        self.cellgrads = scale * (normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
        scalemass = 1 / self.nloc / (self.nloc + 1);
        massloc = np.tile(scalemass, (self.nloc, self.nloc))
        massloc.reshape((self.nloc * self.nloc))[::self.nloc + 1] *= 2
        self.mass = np.einsum('n,kl->nkl', dV, massloc).flatten()

    def boundary(self, A, b, u, bdrycond, bdrydata, method):
        x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        self.bsaved = []
        self.Asaved = []
        for icomp in range(ncomp):
            nodedirall, nodesinner, nodesdir, nodesdirflux = bdrydata[icomp]
            self.bsaved.append({})
            self.Asaved.append({})
            for key, nodes in nodesdirflux.items():
                self.bsaved[icomp][key] = b[icomp + ncomp * nodes]
            for key, nodes in nodesdirflux.items():
                nb = nodes.shape[0]
                help = sparse.dok_matrix((nb, ncomp * nnodes))
                for i in range(nb): help[i, icomp + ncomp * nodes[i]] = 1
                self.Asaved[icomp][key] = help.dot(A)
            if method == 'trad':
                for color, nodes in nodesdir.items():
                    dirichlet = bdrycond[icomp].fct[color]
                    b[icomp + ncomp * nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                    u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
                indin = icomp + ncomp *nodesinner
                inddir = icomp + ncomp *nodedirall
                b[indin] -= A[indin, :][:,inddir] * b[inddir]
                help = np.ones((ncomp * nnodes))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                A = help.dot(A.dot(help))
                help = np.zeros((ncomp * nnodes))
                help[inddir] = 1.0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                A += help
            else:
                for color, nodes in nodesdir.items():
                    dirichlet = bdrycond[icomp].fct[color]
                    u[icomp + ncomp * nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                    b[icomp + ncomp * nodes] = 0
                # print("b", b)
                # print("u", u)
                # b -= A*u
                indin = icomp + ncomp *nodesinner
                inddir = icomp + ncomp *nodedirall
                b[indin] -= A[indin, :][:, inddir] * u[inddir]
                b[inddir] = A[inddir, :][:, inddir] * u[inddir]
                # print("b", b)
                help = np.ones((ncomp * nnodes))
                help[inddir] = 0
                # print("help", help)
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                help2 = np.zeros((ncomp * nnodes))
                help2[inddir] = 1
                # print("help2", help2)
                help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
                # print("A", A)
        return A, b, u

    def tonode(self, u):
        return u

    def grad(self, ic):
        normals = self.mesh.normals[self.mesh.facesOfCells[ic, :]]
        grads = 0.5 * normals / self.mesh.dV[ic]
        chsg = (ic == self.mesh.cellsOfFaces[self.mesh.facesOfCells[ic, :], 0])
        # print("### chsg", chsg, "normals", normals)
        grads[chsg] *= -1.
        return grads

    def phi(self, ic, x, y, z, grad):
        return 1. / 3. + np.dot(grad, np.array(
            [x - self.mesh.pointsc[ic, 0], y - self.mesh.pointsc[ic, 1], z - self.mesh.pointsc[ic, 2]]))

    def testgrad(self):
        for ic in range(fem.mesh.ncells):
            grads = fem.grad(ic)
            for ii in range(3):
                x = self.mesh.points[self.mesh.simplices[ic, ii], 0]
                y = self.mesh.points[self.mesh.simplices[ic, ii], 1]
                z = self.mesh.points[self.mesh.simplices[ic, ii], 2]
                for jj in range(3):
                    phi = self.phi(ic, x, y, z, grads[jj])
                    if ii == jj:
                        test = np.abs(phi - 1.0)
                        if test > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            print('x,y', x, y)
                            print('x-xc,y-yc', x - self.mesh.pointsc[ic, 0], y - self.mesh.pointsc[ic, 1])
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic, ii, jj, test))
                    else:
                        test = np.abs(phi)
                        if np.abs(phi) > 1e-14:
                            print('ic=', ic, 'grad=', grads)
                            raise ValueError('wrong in cell={}, ii,jj={},{} test= {}'.format(ic, ii, jj, test))

    def computeErrorL2(self, solex, uh):
        x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
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
            mean += np.sum(dS * np.mean(u[icomp + self.ncomp * self.mesh.faces[faces]], axis=1))
        return mean

    def computeBdryDn(self, u, key, data, icomp):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[icomp][key] - self.Asaved[icomp][key] * u)
        return flux


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemP1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt

    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
