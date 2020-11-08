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
from ..tools.barycentricmatrix import tensor


#=================================================================#
class P1(object):
    def __init__(self, mesh=None):
        if mesh is not None: self.setMesh(mesh)
        self.dirichlet_al = 10
    def setMesh(self, mesh):
        self.mesh = mesh
        self.nloc = self.mesh.dimension+1
        simps = self.mesh.simplices
        self.cols = np.tile(simps, self.nloc).reshape(-1)
        self.rows = np.repeat(simps, self.nloc).reshape(-1)
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
        # print("dV", np.where(dV<0.0001))
        # print("dV", dV[dV<0.00001])
        return scale*(normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
    def computeMassMatrix(self, coeff=1):
        nnodes = self.mesh.nnodes
        scalemass = 1 / self.nloc / (self.nloc+1);
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] *= 2

        mass = np.einsum('n,kl->nkl', coeff*self.mesh.dV, massloc).ravel()
        return sparse.coo_matrix((mass, (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    def computeBdryMassMatrix(self, bdrycond, bdrycondtype, lumped=False):
        nnodes = self.mesh.nnodes
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        if lumped:
            for color, faces in self.mesh.bdrylabels.items():
                if bdrycond.type[color] != bdrycondtype: continue
                scalemass = bdrycond.param[color]/ self.mesh.dimension
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                nodes = self.mesh.faces[faces]
                rows = np.append(rows, nodes)
                cols = np.append(cols, nodes)
                mass = np.repeat(scalemass*dS, self.mesh.dimension)
                mat = np.append(mat, mass)
        else:
            for color, faces in self.mesh.bdrylabels.items():
                if bdrycond.type[color] != bdrycondtype: continue
                scalemass = bdrycond.param[color] / (1+self.mesh.dimension)/self.mesh.dimension
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                nodes = self.mesh.faces[faces]
                nloc = self.nloc-1
                rows = np.append(rows, np.repeat(nodes, nloc).reshape(-1))
                cols = np.append(cols, np.tile(nodes, nloc).reshape(-1))
                massloc = np.tile(scalemass, (nloc, nloc))
                massloc.reshape((nloc*nloc))[::nloc+1] *= 2
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).reshape(-1))
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nnodes, nnodes)).tocsr()
    def matrixDiffusion(self, coeff):
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

    def computeErrorL2(self, solexact, uh):
        xc, yc, zc = self.mesh.pointsc.T
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.simplices], axis=1)
        return np.sqrt(np.sum(ec**2* self.mesh.dV)), ec

    def computeErrorL2Mass(self, solexact, uh, mass):
        x, y, z = self.mesh.points.T
        en = solexact(x, y, z) - uh
        return np.sqrt( np.dot(en, mass*en) ), en

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
