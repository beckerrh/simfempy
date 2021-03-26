# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from simfempy.fems import femsys, p1

#=================================================================#
class P1sys(femsys.Femsys):
    def __init__(self, ncomp, mesh=None):
        super().__init__(p1.P1(mesh), ncomp, mesh)
    def computeRhs(self, problemdata):
        ncomp = self.ncomp
        b = np.zeros(self.mesh.nnodes * self.ncomp)
        rhs = problemdata.params.fct_glob['rhs']
        if rhs:
            rhsall = self.fem.interpolate(rhs)
            for i in range(ncomp):
                self.fem.massDot(b[i::self.ncomp], rhsall[i])
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
                indices = i + self.ncomp * self.mesh.faces[faces]
                np.add.at(b, indices.T, bS)
        return b
    def vectorDirichlet(self, problemdata, method, b, u):
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        x, y, z = self.mesh.points.T
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodesdir, nodedirall, nodesinner, nodesdirflux = self.bdrydata.nodesdir, self.bdrydata.nodedirall, self.bdrydata.nodesinner, self.bdrydata.nodesdirflux
        # print(f"vectorDirichlet {nodesdirflux.items()=}")
        for key, nodes in nodesdirflux.items():
            ind = np.repeat(ncomp * nodes, ncomp)
            for icomp in range(ncomp): ind[icomp::ncomp] += icomp
            self.bdrydata.bsaved[key] = b[ind]
        indin = np.repeat(ncomp * nodesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * nodedirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        if method == 'trad':
            for color, nodes in nodesdir.items():
                if color in problemdata.bdrycond.fct.keys():
                    dirichlets = problemdata.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = dirichlets[icomp]
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = 0
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
            b[indin] -= self.bdrydata.A_inner_dir * b[inddir]
        else:
            for color, nodes in nodesdir.items():
                if color in problemdata.bdrycond.fct.keys():
                    dirichlets = problemdata.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    for icomp in range(ncomp):
                        u[icomp + ncomp * nodes] = dirichlets[icomp]
                        b[icomp + ncomp * nodes] = 0
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = 0
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
            b[indin] -= self.bdrydata.A_inner_dir * u[inddir]
            b[inddir] = self.bdrydata.A_dir_dir * u[inddir]
        # print(f"vectorDirichlet {self.bdrydata.bsaved.keys()=}")
        return b, u
    def computeMatrixElasticity(self, mucell, lamcell):
        nnodes, ncells, ncomp, dV = self.mesh.nnodes, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, rows, cols, cellgrads = self.fem.nloc, self.rowssys, self.colssys, self.fem.cellgrads
        mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        for i in range(ncomp):
            for j in range(self.ncomp):
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, j]).T * dV * lamcell).T
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, i]).T * dV * mucell).T
                mat[:, i::ncomp, i::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, j]).T * dV * mucell).T
        A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nnodes, ncomp*nnodes)).tocsr()
        return A
    def matrixDirichlet(self, method, A):
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodesdir, nodedirall, nodesinner, nodesdirflux = self.bdrydata.nodesdir, self.bdrydata.nodedirall, self.bdrydata.nodesinner, self.bdrydata.nodesdirflux
        for key, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((ncomp *nb, ncomp * nnodes))
            for icomp in range(ncomp):
                for i in range(nb): help[icomp + ncomp * i, icomp + ncomp * nodes[i]] = 1
            self.bdrydata.Asaved[key] = help.dot(A)
        indin = np.repeat(ncomp * nodesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * nodedirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        self.bdrydata.A_inner_dir = A[indin, :][:, inddir]
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
            self.bdrydata.A_dir_dir = A[inddir, :][:, inddir]
            help = np.ones((ncomp * nnodes))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            help2 = np.zeros((ncomp * nnodes))
            help2[inddir] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A
    def computeBdryNormalFlux(self, u, colors):
        flux, omega = np.zeros(shape=(len(colors),self.ncomp)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if color not in self.bdrydata.bsaved.keys():
                raise KeyError(f"given {color} but known keys {self.bdrydata.bsaved.keys()} {self.bdrydata.Asaved.keys()}")
            bs, As = self.bdrydata.bsaved[color], self.bdrydata.Asaved[color]
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
