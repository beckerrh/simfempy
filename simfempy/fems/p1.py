# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from ..meshes.simplexmesh import SimplexMesh
import simfempy.fems.bdrydata
import simfempy.tools.barycentric
from . import fem


#=================================================================#
class P1(fem.Fem):
    def __init__(self, mesh=None):
        super().__init__(mesh)
        self.dirichlet_al = 10
    def setMesh(self, mesh):
        super().setMesh(mesh)
        nloc = self.mesh.dimension+1
        simps = self.mesh.simplices
        self.cols = np.tile(simps, nloc).reshape(-1)
        self.rows = np.repeat(simps, nloc).reshape(-1)
        self.cellgrads = self.computeCellGrads()
    def nunknowns(self): return self.mesh.nnodes
    def prepareBoundary(self, colorsdir, colorsflux):
        bdrydata = simfempy.fems.bdrydata.BdryData()
        bdrydata.nodesdir={}
        bdrydata.nodedirall = np.empty(shape=(0), dtype=int)
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.nodesdir[color] = np.unique(self.mesh.faces[facesdir].flat[:])
            bdrydata.nodedirall = np.unique(np.union1d(bdrydata.nodedirall, bdrydata.nodesdir[color]))
        # print(f"{bdrydata.nodesdir=}")
        bdrydata.nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=int),bdrydata.nodedirall)
        bdrydata.nodesdirflux={}
        for color in colorsflux:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.nodesdirflux[color] = np.unique(self.mesh.faces[facesdir].ravel())
        return bdrydata
    def computeCellGrads(self):
        ncells, normals, cellsOfFaces, facesOfCells, dV = self.mesh.ncells, self.mesh.normals, self.mesh.cellsOfFaces, self.mesh.facesOfCells, self.mesh.dV
        scale = -1/self.mesh.dimension
        return scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
    def interpolate(self, f):
        x, y, z = self.mesh.points.T
        return f(x, y, z)
    def interpolateCell(self, f):
        if isinstance(f, dict):
            b = np.zeros(self.mesh.ncells)
            for label, fct in f.items():
                if fct is None: continue
                cells = self.mesh.cellsoflabel[label]
                xc, yc, zc = self.mesh.pointsc[cells].T
                b[cells] = fct(xc, yc, zc)
            return b
        else:
            xc, yc, zc = self.mesh.pointsc.T
            return f(xc, yc, zc)
    def interpolateBoundary(self, colors, f):
        b = np.zeros(self.mesh.nnodes)
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            nodes = np.unique(self.mesh.faces[faces].reshape(-1))
            x, y, z = self.mesh.points[nodes].T
            # constant normal !!
            nx, ny, nz = np.mean(normalsS, axis=0)
            b[nodes] = f[color](x, y, z, nx, ny, nz)
        return b

    def massDotCell(self, b, f, coeff=1):
        assert f.shape[0] == self.mesh.ncells
        dimension, simplices, dV = self.mesh.dimension, self.mesh.simplices, self.mesh.dV
        massloc = simfempy.tools.barycentric.integral(d=dimension)
        np.add.at(b, simplices, massloc*coeff*dV*f)
        return b
    def massDot(self, b, f, coeff=1):
        dimension, simplices, dV = self.mesh.dimension, self.mesh.simplices, self.mesh.dV
        massloc = simfempy.tools.barycentric.tensor(d=dimension, k=2)
        r = np.einsum('n,kl,nl->nk', coeff*dV, massloc, f[simplices])
        np.add.at(b, simplices, r)
        return b
    def computeMassMatrix(self, coeff=1):
        dimension, dV, nnodes = self.mesh.dimension, self.mesh.dV, self.mesh.nnodes
        massloc = simfempy.tools.barycentric.tensor(d=dimension, k=2)
        mass = np.einsum('n,kl->nkl', coeff*dV, massloc).ravel()
        return sparse.coo_matrix((mass, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    def massDotBoundary(self, b, f, colors, coeff=1, lumped=False):
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            if isinstance(color, (int,float)): scalemass = coeff
            else: scalemass = coeff[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            nodes = self.mesh.faces[faces]
            massloc = scalemass * simfempy.tools.barycentric.tensor(d=self.mesh.dimension-1, k=2)
            r = np.einsum('n,kl,nl->nk', dS, massloc, f[nodes])
            np.add.at(b, nodes, r)
        return b
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        nnodes = self.mesh.nnodes
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        if colors is None: colors = self.mesh.bdrylabels.keys()
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            if isinstance(coeff, dict):
                scalemass = coeff[color]
                dS = linalg.norm(normalsS, axis=1)
            else:
                scalemass = 1
                dS = linalg.norm(normalsS, axis=1)*coeff[faces]
            nodes = self.mesh.faces[faces]
            if lumped:
                scalemass /= self.mesh.dimension
                rows = np.append(rows, nodes)
                cols = np.append(cols, nodes)
                mass = np.repeat(scalemass * dS, self.mesh.dimension)
                mat = np.append(mat, mass)
            else:
                nloc = self.mesh.dimension
                rows = np.append(rows, np.repeat(nodes, nloc).reshape(-1))
                cols = np.append(cols, np.tile(nodes, nloc).reshape(-1))
                massloc = scalemass * simfempy.tools.barycentric.tensor(d=self.mesh.dimension-1, k=2)
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).reshape(-1))
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nnodes, nnodes)).tocsr()
    def comptuteMatrixTransport(self, beta, betaC, ld):
        nnodes, ncells, nfaces, dim = self.mesh.nnodes, self.mesh.ncells, self.mesh.nfaces, self.mesh.dimension
        if beta.shape != (nfaces,): raise TypeError(f"beta has wrong dimension {beta.shape=} expected {nfaces=}")
        if ld.shape != (ncells, dim+1): raise TypeError(f"ld has wrong dimension {ld.shape=}")
        nlocal = dim+1
        fofc = self.mesh.facesOfCells
        mat = np.einsum('njk,nk,ni -> nij', self.cellgrads[:,:,:dim], betaC, ld)
        A =  sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
        A += self.computeBdryMassMatrix(coeff=np.maximum(beta,0))
        return A

    def computematrixDiffusion(self, coeff):
        nnodes = self.mesh.nnodes
        matxx = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 0], self.cellgrads[:, :, 0])
        matyy = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 1], self.cellgrads[:, :, 1])
        matzz = np.einsum('nk,nl->nkl', self.cellgrads[:, :, 2], self.cellgrads[:, :, 2])
        mat = ( (matxx+matyy+matzz).T*self.mesh.dV*coeff).T.ravel()
        return  sparse.coo_matrix((mat, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    def formDiffusion(self, du, u, coeff):
        graduh = np.einsum('nij,ni->nj', self.cellgrads, u[self.mesh.simplices])
        graduh = np.einsum('ni,n->ni', graduh, self.mesh.dV*coeff)
        # du += np.einsum('nj,nij->ni', graduh, self.cellgrads)
        raise ValueError(f"graduh {graduh.shape} {du.shape}")
        return du
    def computeRhsMass(self, b, rhs, mass):
        if rhs is None: return b
        x, y, z = self.mesh.points.T
        b += mass * rhs(x, y, z)
        return b
    def computeRhsCell(self, b, rhscell):
        if rhscell is None: return b
        scale = 1 / (self.mesh.dimension + 1)
        for label, fct in rhscell.items():
            if fct is None: continue
            cells = self.mesh.cellsoflabel[label]
            xc, yc, zc = self.mesh.pointsc[cells].T
            bC = scale * fct(xc, yc, zc) * self.mesh.dV[cells]
            # print("bC", bC)
            np.add.at(b, self.mesh.simplices[cells].T, bC)
        return b
    def computeRhsPoint(self, b, rhspoint):
        if rhspoint is None: return b
        for label, fct in rhspoint.items():
            if fct is None: continue
            points = self.mesh.verticesoflabel[label]
            xc, yc, zc = self.mesh.points[points].T
            # print("xc, yc, zc, f", xc, yc, zc, fct(xc, yc, zc))
            b[points] += fct(xc, yc, zc)
        return b
    def computeRhsBoundary(self, b, bdrycond, types):
        normals =  self.mesh.normals
        scale = 1 / self.mesh.dimension
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] not in types: continue
            if not color in bdrycond.fct or bdrycond.fct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            assert(dS.shape[0] == len(faces))
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            bS = scale * bdrycond.fct[color](xf, yf, zf, nx, ny, nz) * dS
            np.add.at(b, self.mesh.faces[faces].T, bS)
        return b
    def computeRhsBoundaryMass(self, b, bdrycond, types, mass):
        normals =  self.mesh.normals
        help = np.zeros(self.mesh.nnodes)
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] not in types: continue
            if not color in bdrycond.fct or bdrycond.fct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            assert(dS.shape[0] == len(faces))
            nodes = np.unique(self.mesh.faces[faces].reshape(-1))
            x, y, z = self.mesh.points[nodes].T
            # constant normal !!
            nx, ny, nz = np.mean(normalsS, axis=0)
            help[nodes] = bdrycond.fct[color](x, y, z, nx, ny, nz)
        # print("help", help)
        b += mass*help
        return b
    def matrixBoundary(self, A, method, bdrydata):
        nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata.nodesdir, bdrydata.nodedirall, bdrydata.nodesinner, bdrydata.nodesdirflux
        nnodes = self.mesh.nnodes
        for color, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((nb, nnodes))
            for i in range(nb): help[i, nodes[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[nodesinner, :][:, nodedirall]
        if method == 'trad':
            help = np.ones((nnodes))
            help[nodedirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
            A = help.dot(A.dot(help))
            help = np.zeros((nnodes))
            help[nodedirall] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
            A += help
        else:
            bdrydata.A_dir_dir = self.dirichlet_al*A[nodedirall, :][:, nodedirall]
            help = np.ones(nnodes)
            help[nodedirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
            help2 = np.zeros(nnodes)
            help2[nodedirall] = np.sqrt(self.dirichlet_al)
            help2 = sparse.dia_matrix((help2, 0), shape=(nnodes, nnodes))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A, bdrydata

    def vectorBoundary(self, b, u, bdrycond, method, bdrydata):
        nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata.nodesdir, bdrydata.nodedirall, bdrydata.nodesinner, bdrydata.nodesdirflux
        if u is None: u = np.zeros_like(b)
        elif u.shape != b.shape : raise ValueError("u.shape != b.shape {} != {}".format(u.shape, b.shape))
        x, y, z = self.mesh.points.T
        for color, nodes in nodesdirflux.items():
            bdrydata.bsaved[color] = b[nodes]
        if method == 'trad':
            for color, nodes in nodesdir.items():
                if color in bdrycond.fct:
                    dirichlet = bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    b[nodes] = dirichlet
                else:
                    b[nodes] = 0
                u[nodes] = b[nodes]
            b[nodesinner] -= bdrydata.A_inner_dir * b[nodedirall]
        else:
            for color, nodes in nodesdir.items():
                dirichlet = bdrycond.fct[color]
                if dirichlet:
                    u[nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
                else:
                    u[nodes] = 0
                b[nodes] = 0
            b[nodesinner] -= bdrydata.A_inner_dir * u[nodedirall]
            b[nodedirall] += bdrydata.A_dir_dir * u[nodedirall]
        return b, u, bdrydata

    def vectorBoundaryZero(self, du, bdrydata):
        nodesdir = bdrydata.nodesdir
        for color, nodes in nodesdir.items():
            du[nodes] = 0
        return du

    def tonode(self, u):
        return u

    def computeErrorL2Cell(self, solexact, uh):
        xc, yc, zc = self.mesh.pointsc.T
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.simplices], axis=1)
        return np.sqrt(np.sum(ec**2* self.mesh.dV)), ec

    def computeErrorL2Node(self, solexact, uh):
        x, y, z = self.mesh.points.T
        en = solexact(x, y, z) - uh
        Men = np.zeros_like(en)
        return np.sqrt( np.dot(en, self.massDot(Men,en)) ), en

    def computeErrorFluxL2(self, solexact, diffcell, uh):
        xc, yc, zc = self.mesh.pointsc.T
        graduh = np.einsum('nij,ni->nj', self.cellgrads, uh[self.mesh.simplices])
        errv = 0
        for i in range(self.mesh.dimension):
            solxi = solexact.d(i, xc, yc, zc)
            errv += np.sum( diffcell*(solxi-graduh[:,i])**2* self.mesh.dV)
        return np.sqrt(errv)

    def computeBdryMean(self, u, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*np.mean(u[self.mesh.faces[faces]],axis=1))
        return mean/omega

    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * np.mean(u[self.mesh.faces[faces]], axis=1))
        xf, yf, zf = self.mesh.pointsf[faces].T
        nx, ny, nz = np.mean(self.mesh.normals[faces], axis=0)
        if uR: uRmean =  np.sum(dS * uR(xf, yf, zf, nx, ny, nz))
        else: uRmean=0
        return cR*(uRmean-uhmean)

    def computeBdryNormalFlux(self, u, colors, bdrydata, bdrycond):
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if color in bdrydata.bsaved.keys():
                bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
                flux[i] = np.sum(As * u - bs)
            else:
                flux[i] = self.comuteFluxOnRobin(u, faces, dS, bdrycond.fct[color], bdrycond.param[color])
        return flux

    def computeBdryFct(self, u, colors):
        nodes = np.empty(shape=(0), dtype=int)
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            nodes = np.unique(np.union1d(nodes, self.mesh.faces[faces].ravel()))
        return self.mesh.points[nodes], u[nodes]

    def computePointValues(self, u, colors):
        up = np.empty(len(colors))
        for i,color in enumerate(colors):
            nodes = self.mesh.verticesoflabel[color]
            up[i] = u[nodes]
        return up

    def computeMeanValues(self, u, colors):
        up = np.empty(len(colors))
        for i, color in enumerate(colors):
            up[i] = self.computeMeanValue(u,color)
        return up

    def computeMeanValue(self, u, color):
        cells = self.mesh.cellsoflabel[color]
        # print("umean", np.mean(u[self.mesh.simplices[cells]],axis=1))
        return np.sum(np.mean(u[self.mesh.simplices[cells]],axis=1)*self.mesh.dV[cells])


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
    fem = P1(trimesh)
    fem.testgrad()
    import plotmesh
    import matplotlib.pyplot as plt
    plotmesh.meshWithBoundaries(trimesh)
    plt.show()
