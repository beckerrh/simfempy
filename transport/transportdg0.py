# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse
from scipy import linalg
import matplotlib.pyplot as plt

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    base =  path.dirname(path.dirname(path.abspath(__file__)))
    print('base', base)
    # sys.path.insert(0, base)
    sys.path.append(base)
    print ('sys.path', sys.path)
from tools.analyticalsolution import AnalyticalSolution
from fem.femp12d import FemP12D
from transport import Transport
from mesh.trimesh import TriMesh
from tools.comparerrors import CompareErrors


class TransportDg0(Transport):
    def __init__(self, mesh=None, beta = None, alpha=0.0, dirichlet=None, rhs=None, solexact=None, problem=None):
        Transport.__init__(self, mesh, beta=beta, alpha=alpha, dirichlet=dirichlet, rhs=rhs, solexact=solexact, problem=problem)

    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        bdryedges =  self.mesh.bdryedges

        bcells = self.computeBcells(rhs)
        b = np.zeros(ncells)
        if bcells is not None:
            b += bcells
        for ie in bdryedges:
            bn = self.betan[ie]
            iv0 =  self.mesh.edges[ie,0]
            iv1 =  self.mesh.edges[ie,1]
            xe = 0.5*( self.mesh.x[iv0] +  self.mesh.x[iv1])
            ye = 0.5*( self.mesh.y[iv0] +  self.mesh.y[iv1])
            #da = np.linalg.norm( self.mesh.normals[ie])
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie,0]
                if ic > -1:
                    #print 'inflow edge (-)', ie, 'cell', ic
                    b[ic] -= bn*dirichlet(xe,ye)
            else:
                ic =  self.mesh.cellsOfEdge[ie,1]
                if ic > -1:
                    #print 'inflow edge (+)', ie, 'cell', ic
                    b[ic] += bn*dirichlet(xe,ye)
        return b

    def form(self, du, u):
        self.formReaction(du, u)
        self.formInterior(du, u)
        self.formBoundary(du, u)
        return du
    def formReaction(self, du, u):
        ncells =  self.mesh.ncells
        for i in range(ncells):
          du[i] += self.alpha*u[i]* self.mesh.area[i]
        return du
    def formInterior(self, du, u):
        cellsOfEdge =  self.mesh.cellsOfEdge
        intedges =  self.mesh.intedges
        nintedges = len( self.mesh.intedges)
        ncells =  self.mesh.ncells
        index = np.zeros(4, dtype=int)
        jndex = np.zeros(4, dtype=int)
        for (count, ie) in enumerate(intedges):
            bn = self.betan[ie]
            for ii in range(2):
                for jj in range(2):
                    index[ii + 2 * jj] = cellsOfEdge[ie, ii]
                    jndex[ii + 2 * jj] = cellsOfEdge[ie, jj]
            if bn < 0.0:
                du[index[0]] -= bn*u[jndex[0]]
                du[index[2]] += bn*u[jndex[2]]
            else:
                du[index[1]] -= bn*u[jndex[1]]
                du[index[3]] += bn*u[jndex[3]]
        return du
    def formBoundary(self, du, u):
        ncells =  self.mesh.ncells
        nbdryedges = len( self.mesh.bdryedges)
        bdryedges =  self.mesh.bdryedges
        for (count, ie) in enumerate(bdryedges):
            bn = self.betan[ie]
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie,0]
                if ic > -1:
                    du[ic] -= bn*u[ic]
            else:
                ic =  self.mesh.cellsOfEdge[ie,1]
                if ic > -1:
                    du[ic] += bn*u[ic]
        return du

    def matrix(self, u=None):
        AR = self.matrixReaction()
        AI = self.matrixInterior()
        AB = self.matrixBoundary()
        return (AI + AB + AR).tocsr()
    def matrixReaction(self):
      ncells =  self.mesh.ncells
      index = np.zeros(ncells, dtype=int)
      jndex = np.zeros(ncells, dtype=int)
      A = np.zeros(ncells, dtype=np.float64)
      for i in range(ncells):
        index[i] = i
        jndex[i] = i
        A[i] += self.alpha* self.mesh.area[i]
      return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(ncells, ncells))
    def matrixBoundary(self):
        ncells =  self.mesh.ncells
        nbdryedges = len( self.mesh.bdryedges)
        bdryedges =  self.mesh.bdryedges
        index = np.zeros(nbdryedges, dtype=int)
        jndex = np.zeros(nbdryedges, dtype=int)
        A = np.zeros(nbdryedges, dtype=np.float64)
        for (count, ie) in enumerate(bdryedges):
            bn = self.betan[ie]
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie,0]
                if ic > -1:
                    index[count] = ic
                    jndex[count] = ic
                    A[count] -= bn
            else:
                ic =  self.mesh.cellsOfEdge[ie,1]
                if ic > -1:
                    index[count] = ic
                    jndex[count] = ic
                    A[count] += bn
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(ncells, ncells))
    def matrixInterior(self):
        cellsOfEdge =  self.mesh.cellsOfEdge
        intedges =  self.mesh.intedges
        nintedges = len( self.mesh.intedges)
        ncells =  self.mesh.ncells
        index = np.zeros(4 * nintedges, dtype=int)
        jndex = np.zeros(4 * nintedges, dtype=int)
        A = np.zeros(4 * nintedges, dtype=np.float64)
        for (count, ie) in enumerate(intedges):
            bn = self.betan[ie]
            for ii in range(2):
                for jj in range(2):
                    index[4 * count + ii + 2 * jj] = cellsOfEdge[ie, ii]
                    jndex[4 * count + ii + 2 * jj] = cellsOfEdge[ie, jj]
            if bn < 0.0:
                A[4 * count + 0] -= bn
                A[4 * count + 2] += bn
            else:
                A[4 * count + 1] -= bn
                A[4 * count + 3] += bn
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(ncells, ncells))


# ------------------------------------- #

if __name__ == '__main__':
    run = 'RotStat'
    # run = 'Ramp'
    # run = 'Errors'

    if run == 'Errors':
        methods = {}
        methods['dg0'] = TransportDg0(problem='Analytic_Sinus')
        compareerrors = CompareErrors(methods)
        compareerrors.compare( orders=[1])
    elif run == 'Ramp' or run == 'RotStat':
        trimesh = TriMesh(hnew=0.2)
        transport = TransportDg0(trimesh, problem=run)
        point_data, cell_data, info = transport.solve()
        trimesh.write('transportdg0'+ run + ' .vtk', point_data=point_data, cell_data=cell_data)
