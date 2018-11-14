# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse
import scipy.linalg as linalg

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transportcg1 import TransportCg1
from tools.comparerrors import CompareErrors
from scipy import linalg


class TransportCg1SupgDynamic(TransportCg1):
    """
    """
    def __init__(self, **kwargs):
        if kwargs.has_key('upwind'):
            self.upwind = kwargs.pop('upwind')
        else:
            self.upwind = "supg"
        if kwargs.has_key('coef'):
            self.coef = kwargs.pop('coef')
        else:
            self.coef = "none"
        print self.upwind, self.coef
        TransportCg1.__init__(self, **kwargs)
        if self.finaltime is not None:
            self.nvisusteps = 111
            self.cfl = 1./12.
        # self.fem = FemP12D(self.mesh)
    def setMesh(self, mesh):
        TransportCg1.setMesh(self, mesh)
        # self.mesh = TriMeshWithNodePatches(mesh)
        # beta = np.stack( self.beta(self.mesh.centersx, self.mesh.centersy), axis=-1 )
        ncells = self.mesh.ncells
        elem =  self.mesh.triangles
        betacells = np.zeros( (ncells,2) )
        assert self.beta
        for ic in range(ncells):
            weight = 1.0/3.0
            for ii0 in range(3):
                ii1 = (ii0+1)%3
                x = 0.5 * (self.mesh.x[elem[ic, ii0]] + self.mesh.x[elem[ic, ii1]])
                y = 0.5 * (self.mesh.y[elem[ic, ii0]] + self.mesh.y[elem[ic, ii1]])
                betaxy = self.beta(x,y)
                betacells[ic] += weight*np.stack(betaxy, axis=-1 )
        self.computeDownwindCoeffs(betacells)
        self.computePatchDiff()
    def computeDownwindCoeffs(self, betacells):
        ncells = self.mesh.ncells
        assert betacells.shape == (ncells, 2)
        self.dowindcoefs = np.zeros( (ncells, 3, 3) , dtype=np.float64)
        if self.dt is None:
            self.downwindinfomesh = self.mesh.computeDownwindInfo(betacells)
            for ic in range(ncells):
                i0 = self.downwindinfomesh[0][ic, 0]
                i1 = self.downwindinfomesh[0][ic, 1]
                for ii in range(3):
                    i2 = self.mesh.triangles[ic, ii]
                    if i2 != i0 and i2 != i1:
                        break
                delta = self.downwindinfomesh[1][ic, 0]
                p = self.downwindinfomesh[1][ic, 1]
                for ii in range(3):
                    if self.mesh.triangles[ic,ii]==i0:
                        self.dowindcoefs[ic, 0, ii] = (p - 1.0/3.0)/delta
                        self.dowindcoefs[ic, 1, ii] = self.mesh.area[ic] *p
                    elif self.mesh.triangles[ic, ii] == i1:
                        self.dowindcoefs[ic, 0, ii] = (1- p - 1.0/3.0)/delta
                        self.dowindcoefs[ic, 1, ii] = self.mesh.area[ic] *(1-p)
                    else:
                        self.dowindcoefs[ic, 0, ii] = (- 1.0/3.0)/delta
                        self.dowindcoefs[ic, 1, ii] = 0.0
                if self.coef == "mh":
                    self.dowindcoefs[ic, 1, :] = 0.0
                    for ii in range(3):
                        if self.mesh.triangles[ic, ii] == i0:
                            if p >= 0.5:
                                self.dowindcoefs[ic, 1, ii] = self.mesh.area[ic]
                        elif self.mesh.triangles[ic, ii] == i1:
                            if p <= 0.5:
                                self.dowindcoefs[ic, 1, ii] = self.mesh.area[ic]
                # fq0 = p * max(0, 2.0 / 3.0 - p) * self.mesh.area[ic] / delta
                # fq1 = (1 - p) * max(0, p - 1. / 3.) * self.mesh.area[ic] / delta
                # for ii, patch in self.mesh.patchinfo[i0].iteritems():
                #     if patch[0][0] == i1:
                #         self.patchdiff[i0][ii] = fq0
                # for ii, patch in self.mesh.patchinfo[i1].iteritems():
                #     if patch[0][0] == i0:
                #         self.patchdiff[i1][ii] = fq1
            # print 'self.dowindcoefs', self.dowindcoefs
            return
        else:
            nnodes = self.mesh.nnodes
            elem = self.mesh.triangles
            A = np.zeros(shape=(2, 2), dtype=np.float64)
            b = np.zeros(shape=(2), dtype=np.float64)
            c = np.zeros(shape=(2), dtype=np.float64)
            for ic in range(ncells):
                i0 = elem[ic,0]
                i1 = elem[ic,1]
                i2 = elem[ic,2]
                dx0 = self.mesh.x[i0] - self.mesh.x[i2]
                dy0 = self.mesh.y[i0] - self.mesh.y[i2]
                dx1 = self.mesh.x[i1] - self.mesh.x[i2]
                dy1 = self.mesh.y[i1] - self.mesh.y[i2]
                A[0, 0] = dx0*dx0 + dy0*dy0
                A[0, 1] = dx0*dx1 + dy0*dy1
                A[1, 0] = dx0*dx1 + dy0*dy1
                A[1, 1] = dx1*dx1 + dy1*dy1
                try:
                    A = linalg.inv(A)
                except:
                    raise ValueError('matrix singular')
                b[0] = self.mesh.centersx[ic] +0.5*self.dt*betacells[ic,0] - self.mesh.x[i2]
                b[1] = self.mesh.centersy[ic] +0.5*self.dt*betacells[ic,1] - self.mesh.y[i2]
                c[0] = b[0]*dx0 + b[1]*dy0
                c[1] = b[0]*dx1 + b[1]*dy1
                x = np.dot(A,c)
                if x[0]<0 or x[0] > 1 or x[1] < 0 or x[1] > 1:
                    print 'x', x
                    raise ValueError('wrong coeff')
                b2 = np.array([  x[0] * dx0 + x[1] * dx1 , x[0] * dy0 + x[1] * dy1])
                if scipy.linalg.norm(b-b2) > 1e-15:
                    print 'error in delta', scipy.linalg.norm(x[0]*beta[ic]-b2)
                    print 'beta[ic]', beta[ic]
                    print 'b', b
                    print 'b2', b2
                    print 'x', x
                    assert 0

                delta = 1/np.sqrt(1.0+ np.dot(betacells[ic], betacells[ic]))
                # fct test
                self.dowindcoefs[ic, 1, 0] = x[0]*self.mesh.area[ic]
                self.dowindcoefs[ic, 1, 1] = x[1]*self.mesh.area[ic]
                self.dowindcoefs[ic, 1, 2] = (1 - x[0] - x[1])*self.mesh.area[ic]
                if self.coef == "mh":
                    stop
                # impl
                self.dowindcoefs[ic, 0, 0] = x[0]/delta
                self.dowindcoefs[ic, 0, 1] = x[1]/delta
                self.dowindcoefs[ic, 0, 2] = (1-x[0]-x[1])/delta
                # expl
                self.dowindcoefs[ic, 2, 0] = (2./3. - x[0]) / delta
                self.dowindcoefs[ic, 2, 1] = (2./3. - x[1]) / delta
                self.dowindcoefs[ic, 2, 2] = (x[0]+x[1] - 1.0/3.0) / delta
        self.dowindcoefs = np.around(self.dowindcoefs, 16) + 0
    def computePatchDiff(self):
        ncells = self.mesh.ncells
        elem = self.mesh.triangles
        for ic in range(ncells):
            for ii in range(3):
                i = elem[ic, ii]
                for jj in range(3):
                    j = elem[ic, jj]
                    for iii, patch in self.mesh.patchinfo[i].iteritems():
                        if patch[0][0] == j:
                            faq = max(0.0, self.dowindcoefs[ic, 1, ii]*self.dowindcoefs[ic, 0, jj])
                            self.patchdiff[i][iii] = max(faq, self.patchdiff[i][iii])
        for i in range(self.mesh.nnodes):
            self.patchdiff[i] = np.around(self.patchdiff[i], 16) + 0
            # print self.patchdiff[i]
        if self.finaltime:
            self.patchdiffexplicit = {}
            for iv in range( self.mesh.nnodes):
                npatch = len(self.mesh.patchinfo[iv])
                self.patchdiffexplicit[iv] = np.zeros(npatch, dtype=np.float64)
            for ic in range(ncells):
                for ii in range(3):
                    i = elem[ic, ii]
                    for jj in range(3):
                        j = elem[ic, jj]
                        for iii, patch in self.mesh.patchinfo[i].iteritems():
                            if patch[0][0] == j:
                                faq = min(0.0, self.dowindcoefs[ic, 1, ii]*(self.dowindcoefs[ic, 2, jj])-1)
                                self.patchdiffexplicit[i][iii] += faq
            # for iv in range( self.mesh.nnodes):
            #     print self.patchdiffexplicit[iv]

    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        bdryedges =  self.mesh.bdryedges
        # right-hand-side
        bcells = self.computeBcells(rhs)
        b = np.zeros(nnodes)
        if bcells is not None:
            for ic in range(ncells):
                for ii in range(3):
                    b[elem[ic,ii]] += self.dowindcoefs[ic, 1, ii]*bcells[ic]/self.mesh.area[ic]
        # Dirichlet
        for ie in bdryedges:
            bn = self.betan[ie]
            iv0 =  self.mesh.edges[ie, 0]
            iv1 =  self.mesh.edges[ie, 1]
            xv0 =  self.mesh.x[iv0]
            xv1 =  self.mesh.x[iv1]
            yv0 =  self.mesh.y[iv0]
            yv1 =  self.mesh.y[iv1]
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie, 0]
                if ic > -1:
                    # print 'inflow edge (-)', ie, 'cell', ic, bn
                    b[iv0] -= 0.5 * bn * dirichlet(xv0, yv0)
                    b[iv1] -= 0.5 * bn * dirichlet(xv1, yv1)
            else:
                ic =  self.mesh.cellsOfEdge[ie, 1]
                if ic > -1:
                    # print 'inflow edge (+)', ie, 'cell', ic, bn
                    b[iv0] += 0.5 * bn * dirichlet(xv0, yv0)
                    b[iv1] += 0.5 * bn * dirichlet(xv1, yv1)
        return b
    def form(self, du, u):
        self.formCell(du, u)
        # if self.finaltime:
        #     self.formPatchesDiffusionTwoDyn(du, u, self.umoinsun, self.umoinsdeux)
        if self.upwind == "sl":
            self.formPatchesDiffusionTwo(du, u, self.patchdiff)
        elif self.upwind == "slsym":
            self.formPatchesDiffusionTwoSym(du, u)
        self.formBoundary(du, u)
        return du
    def matrix(self, u=None):
        AI = self.matrixCell()
        # if self.finaltime:
        #     AI += self.matrixPatchesDiffusionTwoDyn(u, self.umoinsun, self.umoinsdeux)
        if self.upwind == "sl":
            AI += self.matrixPatchesDiffusionTwo(u)
            # AI += self.matrixPatchesDiffusionTwoScockCapturing(u)
            # AI += self.matrixPatchesDiffusionLinear()
        elif self.upwind == "slsym":
            AI += self.matrixPatchesDiffusionTwoSym(u)
        AB = self.matrixBoundary()
        A = (AI + AB).tocsr()
        # self.checkMMatrix(A)
        return A
    def formCell(self, du, u):
        ncells =  self.mesh.ncells
        elem =  self.mesh.triangles
        for ic in range(ncells):
            ubeta = np.dot(self.dowindcoefs[ic, 0,:], u[elem[ic,:]])
            for ii in range(3):
                du[elem[ic,ii]] += self.dowindcoefs[ic, 1, ii]*ubeta
        du = np.around(du, 16) + 0
        return du
    def matrixCell(self):
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        index = np.zeros(9 * ncells, dtype=int)
        jndex = np.zeros(9 * ncells, dtype=int)
        A = np.zeros(9 * ncells, dtype=np.float64)
        count=0
        for ic in range(ncells):
            for ii in range(3):
                for jj in range(3):
                    index[count + 3 * ii + jj] = elem[ic, ii]
                    jndex[count + 3 * ii + jj] = elem[ic, jj]
                    A[count + 3 * ii + jj] += self.dowindcoefs[ic, 1, ii]*self.dowindcoefs[ic, 0, jj]
            count += 9
        A = np.around(A, 16) + 0
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))


    def computeRhsDynamic(self, uold):
        self.b[:]= 0.0
        ncells = self.mesh.ncells
        elem =  self.mesh.triangles
        for ic in range(ncells):
            ubeta = np.dot(self.dowindcoefs[ic, 2, :], uold[elem[ic, :]])
            for ii in range(3):
                self.b[elem[ic, ii]] += self.dowindcoefs[ic, 1, ii] * ubeta
        self.b = np.around(self.b, 16) + 0
        # self.computePatchDiff(uold)
        self.A = self.matrixDynamic(uold)
        # if self.upwind == "sl":
        #     # self.formPatchesDiffusionTwoLinear(self.b, uold)
        #     self.formPatchesDiffusionTwo(self.b, uold)
        return self.b

    def residualDynamic(self, u):
        self.r[:]=0
        self.formCell(self.r, u)
        self.formBoundary(self.r, u)
        if self.upwind == "sl":
            # self.formPatchesDiffusionTwoLinear(self.r, u)
            self.formPatchesDiffusionTwo(self.r, u, self.patchdiff)
        self.r -=  self.b
        return np.around(self.r, 16)

    def matrixDynamic(self, u):
        AI = self.matrixCell()
        if self.upwind == "sl":
            # AI += self.matrixPatchesDiffusionLinear()
            AI += self.matrixPatchesDiffusionTwo(u)
        AB = self.matrixBoundary()
        self.A = (AI + AB).tocsr()
        return self.A


# ------------------------------------- #

if __name__ == '__main__':

    problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    # problem = 'Analytic_Exponential'
    # problem = 'Analytic_Linear'
    # problem = 'RotStat'
    problem = 'Ramp'
    alpha = 1.0
    alpha = 0.0
    # beta = lambda x, y: (-y, x)
    beta = lambda x, y: (-np.cos(np.pi * (x + y)), np.cos(np.pi * (x + y)))
    beta = lambda x, y: (-np.sin(np.pi * x) * np.cos(np.pi * y), np.sin(np.pi * y) * np.cos(np.pi * x))
    beta = lambda x, y: (-np.cos(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * y) * np.sin(np.pi * x))
    beta = None

    methods = {}
    methods['supg'] = TransportCg1SupgDynamic(problem=problem, upwind = "supg")
    # methods['supg2'] = TransportCg1SupgDynamic(problem=problem, upwind = "supg", coef="mh")
    # methods['xi'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi')
    methods['xilin'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xilin', coef="mh")
    # methods['xi2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi', coef="mh")
    # methods['xib'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xibis')
    # methods['xit'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xiter')
    # methods['xiq'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xiquater')
    # methods['xi2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi2')
    # methods['xi3'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi3')
    # methods['xisca3'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi3', shockcapturing="phiabs")
    # methods['xiscc3'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi3', shockcapturing="phicon")
    # methods['xisca'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi', shockcapturing="phiabs")
    # methods['xinew'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xinew')
    # methods['xinew2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xinew2')
    # methods['xisignmin'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xisignmin')
    # methods['xispline'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xispline')

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    niter = 3
    h = [0.2*np.power(0.5,i)  for i in range(niter)]
    compareerrors.compare(h=h)