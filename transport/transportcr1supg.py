# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transportcg1 import TransportCg1
from tools.comparerrors import CompareErrors
from fem.femp12d import FemP12D
import scipy.linalg as linalg

class TransportCr1Supg(TransportCg1):
    """
    We suppose that self.mesh.sidesOfCell[ic, ii] is the edge opposit to ii !!
    psi = 1 - 2 phi
    """
    def __init__(self, **kwargs):
        if kwargs.has_key('delta'):
            self.delta = kwargs.pop('delta')
        else:
            self.delta = 1.0
        if kwargs.has_key('galerkin'):
            self.galerkin = kwargs.pop('galerkin')
        else:
            self.galerkin = 'antisym'
        TransportCg1.__init__(self, **kwargs)
        self.fem = FemP12D(self.mesh)

    def setMesh(self, mesh):
        TransportCg1.setMesh(self, mesh)
        self.fem.setMesh(mesh)

    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        bdryedges =  self.mesh.bdryedges
        # right-hand-side
        bcells = self.computeBcells(rhs)
        b = np.zeros(nedges)

        if bcells is not None:
            # for ic in range(ncells):
            #     for ii in range(3):
            #         ind = self.mesh.edgesOfCell[ic, ii]
            #         # assert  np.isclose(self.mesh.edges[ind],  np.setxor1d(self.mesh.triangles[ic],self.mesh.triangles[ic,ii]))
            #         b[ind] += bcells[ic] / 3.0
            for ic in range(ncells):
                beta = self.beta(self.mesh.centersx[ic], self.mesh.centersy[ic])
                delta = self.delta * np.sqrt( self.mesh.area[ic]) / (np.linalg.norm(beta) + 1e-10)
                gradphi = -2.0 * self.fem.grad(ic)
                betagradphi = np.dot(beta, gradphi.T)
                for ii in range(3):
                    ind = self.mesh.edgesOfCell[ic, ii]
                    (xe, ye) = (self.mesh.edgesx[ind], self.mesh.edgesy[ind])
                    grad = [self.solexact.x(xe, ye), self.solexact.y(xe, ye)]
                    u = self.solexact(xe, ye)
                    fii = self.alpha*u + np.dot(beta, grad)
                    b[ind] += fii*self.mesh.area[ic]*(1.0 + delta * betagradphi[ii] )/3.0
            # for ic in range(ncells):
            #     beta = self.beta(self.mesh.centersx[ic], self.mesh.centersy[ic])
            #     gradphi = self.fem.grad(ic)
            #     betagradphi = -2.0*np.dot(beta, gradphi.T)
            #     delta = self.delta * np.sqrt( self.mesh.area[ic]) / (np.linalg.norm(beta) + 1e-10)
            #     for ii in range(3):
            #         b[self.mesh.edgesOfCell[ic, ii]] += delta * bcells[ic] * betagradphi[ii] / 3.0

        # for ii in range(3):
        #             ie =  self.mesh.edgesOfCell[ic,ii]
        # Dirichlet
        for ie in bdryedges:
            bn = self.betan[ie]
            iv0 =  self.mesh.edges[ie, 0]
            iv1 =  self.mesh.edges[ie, 1]
            xv0 =  self.mesh.x[iv0]
            xv1 =  self.mesh.x[iv1]
            yv0 =  self.mesh.y[iv0]
            yv1 =  self.mesh.y[iv1]
            dir = 0.5*( dirichlet(xv0, yv0) + dirichlet(xv1, yv1) )
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie, 0]
                if ic > -1:
                    # print 'inflow edge (-)', ie, 'cell', ic, bn
                    b[ie] -= bn * dir
            else:
                ic =  self.mesh.cellsOfEdge[ie, 1]
                if ic > -1:
                    # print 'inflow edge (+)', ie, 'cell', ic, bn
                    b[ie] += bn * dir
        # print 'b', b
        return b
    def form(self, du, u):
        self.formReaction(du, u)
        if self.galerkin == 'antisym':
            self.formCellAntisym(du, u)
            self.formBoundaryAntisym(du, u)
        elif self.galerkin == 'primal':
            self.formCellPrimal(du, u)
            self.formBoundaryPrimal(du, u)
        elif self.galerkin == 'dual':
            self.formCellPrimal(du, u)
            self.formBoundaryPrimal(du, u)
        else:
            raise ValueError('unknown galerkin form %s' %self.galerkin)
        return du
    def matrix(self, u=None):
        AR = self.matrixReaction()
        if self.galerkin == 'antisym':
            AI = self.matrixCellAntisym()
            AB = self.matrixBoundaryAntisym()
        elif self.galerkin == 'primal':
            AI = self.matrixCellPrimal()
            AI += self.matrixInteriorPrimal()
            AB = self.matrixBoundaryPrimal()
        elif self.galerkin == 'dual':
            AI = self.matrixCellDual()
            AB = self.matrixBoundaryDual()
        else:
            raise ValueError('unknown galerkin form %s' %self.galerkin)
        A = (AI + AB + AR).tocsr()
        # print 'A', A.toarray()
        return A


    def solve(self):
        return self.solveLinear()

    def formReaction(self, du, u):
      ncells =  self.mesh.ncells
      for ic in range(ncells):
          for ii in range(3):
            ind = self.mesh.edgesOfCell[ic, ii]
            du[ind] += u[ind]*self.alpha* self.mesh.area[ic] / 3.0
      return du
    def formBoundaryPrimal(self, du, u):
        bdryedges =  self.mesh.bdryedges
        for (count, ie) in enumerate(bdryedges):
            bn = self.betan[ie]
            if self.mesh.cellsOfEdge[ie, 0]<0: bn *= -1.0
            if bn < 0.0:
                du[ie] -= bn * u[ie]
        return du
    def formCellPrimal(self, du, u):
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        index = np.zeros(9, dtype=int)
        jndex = np.zeros(9, dtype=int)
        for ic in range(ncells):
            for jj in range(3):
                for ii in range(3):
                    index[ii + 3 * jj] = self.mesh.edgesOfCell[ic, ii]
                    jndex[ii + 3 * jj] = self.mesh.edgesOfCell[ic, jj]
            beta = self.beta(self.mesh.centersx[ic], self.mesh.centersy[ic])
            gradphi = -2.0*self.fem.grad(ic)
            betagradphi = np.dot(beta, gradphi.T)
            # print 'betagradphi=', betagradphi
            for jj in range(3):
                betagradphij = betagradphi[jj] / 3.0 *  self.mesh.area[ic]
                for ii in range(3):
                    du[index[ii + 3 * jj]] += betagradphij * u[jndex[ii + 3 * jj]]
            delta = self.delta * np.sqrt( self.mesh.area[ic]) / (np.linalg.norm(beta) + 1e-10)
            for jj in range(3):
                betagradphij = betagradphi[jj] / 3.0 *  self.mesh.area[ic]
                for ii in range(3):
                    du[index[ii + 3 * jj]] += delta * betagradphij * betagradphi[ii] * u[jndex[ii + 3 * jj]]
        return du
    def matrixReaction(self):
        ncells =  self.mesh.ncells
        nedges = self.mesh.nedges
        index = np.zeros(nedges, dtype=int)
        jndex = np.zeros(nedges, dtype=int)
        A = np.zeros(nedges, dtype=np.float64)
        for ie in range(nedges):
            index[ie] = ie
            jndex[ie] = ie
        for ic in range(ncells):
            for ii in range(3):
                ind = self.mesh.edgesOfCell[ic, ii]
                A[ind] += self.alpha* self.mesh.area[ic] / 3.0
        # print 'A', A
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixBoundaryPrimal(self):
        nedges =  self.mesh.nedges
        nbdryedges = len( self.mesh.bdryedges)
        bdryedges =  self.mesh.bdryedges
        index = np.zeros(nbdryedges, dtype=int)
        jndex = np.zeros(nbdryedges, dtype=int)
        A = np.zeros(nbdryedges, dtype=np.float64)
        for (count, ie) in enumerate(bdryedges):
            index[count] = ie
            jndex[count] = ie
            bn = self.betan[ie]
            if self.mesh.cellsOfEdge[ie, 0]<0: bn *= -1.0
            if bn < 0.0:
                A[count] -= bn
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixCellPrimal(self):
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        elem =  self.mesh.triangles
        index = np.zeros(9 * ncells, dtype=int)
        jndex = np.zeros(9 * ncells, dtype=int)
        A = np.zeros(9 * ncells, dtype=np.float64)
        for ic in range(ncells):
            for jj in range(3):
                for ii in range(3):
                    index[9 * ic + ii + 3 * jj] = self.mesh.edgesOfCell[ic, ii]
                    jndex[9 * ic + ii + 3 * jj] = self.mesh.edgesOfCell[ic, jj]
            xc = np.mean( self.mesh.x[elem[ic, :]])
            yc = np.mean( self.mesh.y[elem[ic, :]])
            beta = self.beta(xc, yc)
            gradphi = -2.0 * self.fem.grad(ic)
            betagradphi = np.dot(beta, gradphi.T)
            # print 'betagradphi=', betagradphi
            for jj in range(3):
                betagradphij =  betagradphi[jj] / 3.0 *  self.mesh.area[ic]
                for ii in range(3):
                    A[9 * ic + ii + 3 * jj] += betagradphij
            delta = self.delta * np.sqrt( self.mesh.area[ic]) / (np.linalg.norm(beta) + 1e-10)
            for jj in range(3):
                betagradphij =  betagradphi[jj] / 3.0 *  self.mesh.area[ic]
                for ii in range(3):
                    A[9 * ic + ii + 3 * jj] += delta * betagradphij * betagradphi[ii]
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixInteriorPrimal(self):
        nedges =  self.mesh.nedges
        nintedges =  self.mesh.nintedges
        nlocal = 8
        index = np.zeros(nlocal * nintedges, dtype=int)
        jndex = np.zeros(nlocal * nintedges, dtype=int)
        A = np.zeros(nlocal * nintedges, dtype=np.float64)
        for (i,ie) in enumerate(self.mesh.intedges):
            h = linalg.norm(self.mesh.normals[ie])
            bn = self.betan[ie]
            ikL = self.mesh.cellsOfEdge[ie,0]
            ikR = self.mesh.cellsOfEdge[ie,1]
            indL = np.setxor1d(self.mesh.edgesOfCell[ikL], ie)
            indR = np.setxor1d(self.mesh.edgesOfCell[ikR], ie)
            indLR = np.hstack( (indL, indR) )
            for jj in range(4):
                for ii in range(2):
                    if bn >= 0.0:
                        index[nlocal*i + 4*ii + jj] = indL[ii]
                    else:
                        index[nlocal * i + 4 * ii + jj] = indR[ii]
                    jndex[nlocal * i + 4 * ii + jj] = indLR[jj]
                    if jj < 2:
                        A[nlocal * i + 4 * ii + jj] += bn/6.0
                    else:
                        A[nlocal * i + 4 * ii + jj] -= bn/6.0
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixBoundaryDual(self):
        nedges = self.mesh.nedges
        nbdryedges = len(self.mesh.bdryedges)
        bdryedges = self.mesh.bdryedges
        index = np.zeros(nbdryedges, dtype=int)
        jndex = np.zeros(nbdryedges, dtype=int)
        A = np.zeros(nbdryedges, dtype=np.float64)
        for (count, ie) in enumerate(bdryedges):
            index[count] = ie
            jndex[count] = ie
            bn = self.betan[ie]
            if self.mesh.cellsOfEdge[ie, 0]<0: bn *= -1.0
            if bn > 0.0:
                A[count] += bn
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixCellDual(self):
        ncells = self.mesh.ncells
        nedges = self.mesh.nedges
        elem = self.mesh.triangles
        index = np.zeros(9 * ncells, dtype=int)
        jndex = np.zeros(9 * ncells, dtype=int)
        A = np.zeros(9 * ncells, dtype=np.float64)
        for ic in range(ncells):
            for jj in range(3):
                for ii in range(3):
                    index[9 * ic + ii + 3 * jj] = self.mesh.edgesOfCell[ic, ii]
                    jndex[9 * ic + ii + 3 * jj] = self.mesh.edgesOfCell[ic, jj]
            xc = np.mean(self.mesh.x[elem[ic, :]])
            yc = np.mean(self.mesh.y[elem[ic, :]])
            beta = self.beta(xc, yc)
            gradphi = -2.0 * self.fem.grad(ic)
            betagradphi = np.dot(beta, gradphi.T)
            # print 'betagradphi=', betagradphi
            for jj in range(3):
                betagradphij = betagradphi[jj] / 3.0 * self.mesh.area[ic]
                for ii in range(3):
                    A[9 * ic + jj + 3 * ii] -= betagradphij
            delta = self.delta * np.sqrt(self.mesh.area[ic]) / (np.linalg.norm(beta) + 1e-10)
            for jj in range(3):
                betagradphij = betagradphi[jj] / 3.0 * self.mesh.area[ic]
                for ii in range(3):
                    A[9 * ic + ii + 3 * jj] += delta * betagradphij * betagradphi[ii]
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixBoundaryAntisym(self):
        nedges =  self.mesh.nedges
        nbdryedges = len( self.mesh.bdryedges)
        bdryedges =  self.mesh.bdryedges
        index = np.zeros(nbdryedges, dtype=int)
        jndex = np.zeros(nbdryedges, dtype=int)
        A = np.zeros(nbdryedges, dtype=np.float64)
        for (count, ie) in enumerate(bdryedges):
            bn = self.betan[ie]
            index[count] = ie
            jndex[count] = ie
            A[count] += 0.5*abs(bn)
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))
    def matrixCellAntisym(self):
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        elem =  self.mesh.triangles
        index = np.zeros(9 * ncells, dtype=int)
        jndex = np.zeros(9 * ncells, dtype=int)
        A = np.zeros(9 * ncells, dtype=np.float64)
        for ic in range(ncells):
            for jj in range(3):
                for ii in range(3):
                    index[9 * ic + ii + 3 * jj] = self.mesh.edgesOfCell[ic, ii]
                    jndex[9 * ic + ii + 3 * jj] = self.mesh.edgesOfCell[ic, jj]
            xc = np.mean( self.mesh.x[elem[ic, :]])
            yc = np.mean( self.mesh.y[elem[ic, :]])
            beta = self.beta(xc, yc)
            gradphi = -2.0 * self.fem.grad(ic)
            betagradphi = np.dot(beta, gradphi.T)
            # print 'betagradphi=', betagradphi
            for jj in range(3):
                betagradphij =  betagradphi[jj] / 3.0 *  self.mesh.area[ic]
                for ii in range(3):
                    A[9 * ic + ii + 3 * jj] += 0.5 * betagradphij
                    A[9 * ic + jj + 3 * ii] -= 0.5 * betagradphij
            delta = self.delta * np.sqrt( self.mesh.area[ic]) / (np.linalg.norm(beta) + 1e-10)
            for jj in range(3):
                betagradphij =  betagradphi[jj] / 3.0 *  self.mesh.area[ic]
                for ii in range(3):
                    A[9 * ic + ii + 3 * jj] += delta * betagradphij * betagradphi[ii]
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges, nedges))


# ------------------------------------- #

if __name__ == '__main__':
    problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    # problem = 'Analytic_Exponential'
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Constant'
    # problem = 'RotStat'
    # problem = 'Ramp'
    alpha = 1.0
    # alpha = 0.0
    # beta = lambda x, y: (-y, x)
    beta = lambda x, y: (-np.cos(np.pi * (x + y)), np.cos(np.pi * (x + y)))
    beta = lambda x, y: (-np.sin(np.pi * x) * np.cos(np.pi * y), np.sin(np.pi * y) * np.cos(np.pi * x))
    beta = lambda x, y: (-np.cos(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * y) * np.sin(np.pi * x))
    beta = lambda x, y: (0,0)
    beta = None

    methods = {}
    # methods['supg'] = TransportCr1Supg(problem=problem, alpha=alpha, beta=beta)
    methods['Cp'] = TransportCr1Supg(delta=0.0, galerkin = 'primal', problem=problem, alpha=alpha, beta=beta)
    # methods['Cd'] = TransportCr1Supg(delta=0.0, galerkin = 'dual', problem=problem, alpha=alpha, beta=beta)
    # methods['Cas'] = TransportCr1Supg(delta=0.0, galerkin = 'antisym', problem=problem, alpha=alpha, beta=beta)

    compareerrors = CompareErrors(methods, latex=True)
    h = [2.0, 1.0]
    from mesh.trimesh import TriMesh
    trimesh = TriMesh(hnew=2)
    import matplotlib.pyplot as plt
    # trimesh.plot(plt, edges=True)
    # h = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03]
    compareerrors.compare(h=h, vtk=True)
