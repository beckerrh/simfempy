# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

from simfempy.fems import femp1

try:
    from simfempy.meshes.simplexmesh import SimplexMesh
except ModuleNotFoundError:
    from ..meshes.simplexmesh import SimplexMesh
import simfempy.fems.bdrydata

# =================================================================#
class FemP1(femp1.FemP1):
    def __init__(self, mesh=None):
        super().__init__(mesh)

    def setMesh(self, mesh, ncomp):
        super().setMesh(mesh)
        self.ncomp = ncomp
        ncells, simps = self.mesh.ncells, self.mesh.simplices
        nlocncomp = ncomp * self.nloc
        self.rowssys = np.repeat(ncomp * simps, ncomp).reshape(ncells * self.nloc, ncomp) + np.arange(ncomp)
        self.rowssys = self.rowssys.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.colssys = self.rowssys.swapaxes(1, 2)
        self.colssys = self.colssys.reshape(-1)
        self.rowssys = self.rowssys.reshape(-1)

    def prepareBoundary(self, bdrycond, postproc):
        if not isinstance(bdrycond, (list, tuple)):
            return super().prepareBoundary(bdrycond.colorsOfType("Dirichlet"), postproc)
        bdrydata = []
        for icomp in range(self.ncomp):
            bdrydata.append(super().prepareBoundary(bdrycond[icomp].colorsOfType("Dirichlet"), postproc[icomp]))
        return bdrydata

    def computeRhs(self, u, rhs, diff, bdrycond, method, bdrydata):
        b = np.zeros(self.mesh.nnodes * self.ncomp)
        if rhs:
            for icomp in range(self.ncomp):
                x, y, z = self.mesh.points.T
                bnodes = rhs[icomp](x, y, z, diff[icomp][0])
                b[icomp::self.ncomp] = self.massmatrix * bnodes
        normals = self.mesh.normals
        scale = 1 / self.mesh.dimension
        for color, faces in self.mesh.bdrylabels.items():
            for icomp in range(self.ncomp):
                if bdrycond[icomp].type[color] != "Neumann": continue
                normalsS = normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                kS = diff[icomp][self.mesh.cellsOfFaces[faces, 0]]
                x1, y1, z1 = self.mesh.pointsf[faces].T
                nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
                bS = scale * dS * bdrycond[icomp].fct[color](x1, y1, z1, nx, ny, nz, kS)
                indices = icomp + self.ncomp * self.mesh.faces[faces]
                np.add.at(b, indices.T, bS)
        return self.vectorDirichlet(b, u, bdrycond, method, bdrydata)

    def matrixDiffusion(self, k, bdrycond, method, bdrydata):
        nnodes, ncells, ncomp = self.mesh.nnodes, self.mesh.ncells, self.ncomp
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        nlocncomp = ncomp * self.nloc
        mat = np.zeros(shape=(ncells, nlocncomp, nlocncomp))
        for icomp in range(ncomp):
            mat[:, icomp::ncomp, icomp::ncomp] = ((matxx + matyy + matzz).T * self.mesh.dV * k[icomp]).T
        A = sparse.coo_matrix((mat.ravel(), (self.rowssys, self.colssys)), shape=(ncomp*nnodes, ncomp*nnodes)).tocsr()
        return self.matrixDirichlet(A, bdrycond, method, bdrydata)

    def vectorDirichlet(self, b, u, bdrycond, method, bdrydata):
        x, y, z = self.mesh.points.T
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        for icomp in range(ncomp):
            nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata[icomp].nodesdir, bdrydata[icomp].nodedirall, bdrydata[icomp].nodesinner, bdrydata[icomp].nodesdirflux
            for key, nodes in nodesdirflux.items():
                bdrydata[icomp].bsaved[key] = b[icomp + ncomp * nodes]
            if method == 'trad':
                for color, nodes in nodesdir.items():
                    dirichlet = bdrycond[icomp].fct[color]
                    b[icomp + ncomp * nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                    u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
                indin = icomp + ncomp *nodesinner
                inddir = icomp + ncomp *nodedirall
                b[indin] -= bdrydata[icomp].A_inner_dir * b[inddir]
            else:
                for color, nodes in nodesdir.items():
                    dirichlet = bdrycond[icomp].fct[color]
                    u[icomp + ncomp * nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                    b[icomp + ncomp * nodes] = 0
                indin = icomp + ncomp *nodesinner
                inddir = icomp + ncomp *nodedirall
                b[indin] -= bdrydata[icomp].A_inner_dir * u[inddir]
                b[inddir] = bdrydata[icomp].A_dir_dir * u[inddir]
        return b, u, bdrydata

    def matrixDirichlet(self, A, bdrycond, method, bdrydata):
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        for icomp in range(ncomp):
            nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata[icomp].nodesdir, bdrydata[icomp].nodedirall, bdrydata[icomp].nodesinner, bdrydata[icomp].nodesdirflux
            indin = icomp + ncomp * nodesinner
            inddir = icomp + ncomp * nodedirall
            bdrydata[icomp].A_inner_dir = A[indin, :][:, inddir]
            for key, nodes in nodesdirflux.items():
                nb = nodes.shape[0]
                help = sparse.dok_matrix((nb, ncomp * nnodes))
                for i in range(nb): help[i, icomp + ncomp * nodes[i]] = 1
                bdrydata[icomp].Asaved[key] = help.dot(A)
            if method == 'trad':
                help = np.ones((ncomp * nnodes))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                A = help.dot(A.dot(help))
                help = np.zeros((ncomp * nnodes))
                help[inddir] = 1.0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                A += help
            else:
                bdrydata[icomp].A_dir_dir = A[inddir, :][:, inddir]
                help = np.ones((ncomp * nnodes))
                help[inddir] = 0
                help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                help2 = np.zeros((ncomp * nnodes))
                help2[inddir] = 1
                help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nnodes, ncomp * nnodes))
                A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, bdrydata

    def boundary(self, A, b, u, bdrycond, bdrydata, method):
        x, y, z = self.mesh.points.T
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
        x, y, z = self.mesh.points.T
        e = []
        err = []
        for icomp in range(self.ncomp):
            e.append(solex[icomp](x, y, z) - uh[icomp::self.ncomp])
            err.append(np.sqrt(np.dot(e[icomp], self.massmatrix * e[icomp])))
        return err, e

    def computeBdryMean(self, u, data, icomp=None):
        colors = [int(x) for x in data.split(',')]
        if icomp is None:
            mean, omega = np.zeros(shape=(self.ncomp,len(colors))), np.zeros(len(colors))
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                for icomp in range(self.ncomp):
                    mean[icomp, i] = np.sum(dS * np.mean(u[icomp + self.ncomp * self.mesh.faces[faces]], axis=1))
            return mean / omega
        else:
            mean, omega = np.zeros(shape=(len(colors))), np.zeros(len(colors))
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                mean[i] = np.sum(dS * np.mean(u[icomp + self.ncomp * self.mesh.faces[faces]], axis=1))
            return mean/omega

    def computeBdryDn(self, u, data, bdrydata, bdrycond, icomp=None):
        colors = [int(x) for x in data.split(',')]
        if icomp is None:
            flux, omega = np.zeros(shape=(self.ncomp,len(colors))), np.zeros(len(colors))
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                for icomp in range(self.ncomp):
                    if bdrycond[icomp].type[color] == "Dirichlet":
                        bs, As = bdrydata[icomp].bsaved[color], bdrydata[icomp].Asaved[color]
                        flux[icomp, i] = np.sum(As * u - bs)
                else:
                    raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
            return flux
        else:
            flux, omega = np.zeros(shape=(len(colors))), np.zeros(len(colors))
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                omega[i] = np.sum(dS)
                if bdrycond[icomp].type[color] == "Dirichlet":
                    bs, As = bdrydata[icomp].bsaved[color], bdrydata[icomp].Asaved[color]
                    flux[i] = np.sum(As * u - bs)
                else:
                    raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
            return flux


    def computePointValues(self, u, data):
        colors = [int(x) for x in data.split(',')]
        up = np.empty(shape=(len(colors), self.ncomp))
        for i,color in enumerate(colors):
            nodes = self.mesh.verticesoflabel[color]
            for icomp in range(self.ncomp):
                up[i, icomp] = u[icomp + self.ncomp *nodes]
        return up

    def computeMeanValues(self, u, data):
        colors = [int(x) for x in data.split(',')]
        up = np.empty(shape=(len(colors), self.ncomp))
        for i, color in enumerate(colors):
            cells = self.mesh.cellsoflabel[color]
            simpcells = self.mesh.simplices[cells]
            for icomp in range(self.ncomp):
                mean = np.sum(np.mean(u[icomp + self.ncomp * simpcells], axis=1) * self.mesh.dV[cells])
                up[i, icomp] = mean
        return up

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = FemP1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt

    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
