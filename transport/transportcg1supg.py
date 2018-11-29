# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse
import scipy.linalg as linalg
import matplotlib.pyplot as plt

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transportcg1 import TransportCg1
from tools.comparerrors import CompareErrors
from scipy import linalg


class TransportCg1Supg(TransportCg1):
    """
    """
    def __init__(self, **kwargs):
        if 'upwind' in kwargs:
            self.upwind = kwargs.pop('upwind')
        else:
            self.upwind = "supg"
        if 'coef' in kwargs:
            self.coef = kwargs.pop('coef')
        else:
            self.coef = "none"
        print(self.upwind, self.coef)
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
        self.computeStencil()
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
                    print('x', x)
                    raise ValueError('wrong coeff')
                b2 = np.array([  x[0] * dx0 + x[1] * dx1 , x[0] * dy0 + x[1] * dy1])
                if scipy.linalg.norm(b-b2) > 1e-15:
                    print('error in delta', scipy.linalg.norm(x[0]*beta[ic]-b2))
                    print('beta[ic]', beta[ic])
                    print('b', b)
                    print('b2', b2)
                    print('x', x)
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
    def computeStencil(self):
        self.stencil = {}
        nnpatch=0
        for iv in range( self.mesh.nnodes):
            npatch = len(self.mesh.patchinfo[iv])
            nnpatch += npatch
            self.stencil[iv] = np.zeros(npatch, dtype=np.float64)
        ncells = self.mesh.ncells
        elem = self.mesh.triangles
        Ac = np.zeros((3, 3), dtype=np.float64)
        for ic in range(ncells):
            for ii in range(3):
                for jj in range(3):
                    Ac[ii,jj] = self.dowindcoefs[ic, 1, ii]*self.dowindcoefs[ic, 0, jj]
            for ii in range(3):
                i = elem[ic, ii]
                for iii, patch in self.mesh.patchinfo[i].items():
                    j = patch[0][0]
                    for jj in range(3):
                        if j == elem[ic, jj]:
                            # self.stencil[i][iii] += Ac[ii,jj]
                            self.stencil[i][iii] += self.dowindcoefs[ic, 1, ii]*self.dowindcoefs[ic, 0, jj]

    def computeShockCapturing(self, u):
        self.sc[:] = 1.0
        return
        for iv in range(self.mesh.nnodes):
            boundary = -1 in self.mesh.patches_bnodes[iv][:, 5]
            if boundary:
                self.sc[iv] = 0.0
            else:
                self.sc[iv] = 0.0
                g, g2 = self.computePatchGradients(iv, u)
                for iii, patch in self.mesh.patchinfo[iv].items():
                    jv = patch[0][0]
                    self.sc[iv] += self.stencil[iv][iii]*g2[iii]
                self.sc[iv]= abs(self.sc[iv])

    def computePatchDiff(self):
        for i in range(self.mesh.nnodes):
            boundary = -1 in self.mesh.patches_bnodes[i][:, 5]
            if boundary:
                continue
            for iii, patchi in self.mesh.patchinfo[i].items():
                j = patchi[0][0]
                boundary = -1 in self.mesh.patches_bnodes[j][:, 5]
                if boundary:
                    continue
                bij = self.stencil[i][iii]
                found = False
                for jjj, patchj in self.mesh.patchinfo[j].items():
                    if patchj[0][0] ==i:
                        bji = self.stencil[j][jjj]
                        found = True
                        break
                assert found
                bijm = min(bij, 0.0)
                bjip = max(bji, 0.0)
                if bijm-bjip > 0.0:
                    print('bij, bji', bij, bji, i, j)
                    self.mesh.plotNodePatches(plt)
                    sys.exit(1)

        for iv in range(self.mesh.nnodes):
            for iii, patch in self.mesh.patchinfo[iv].items():
                self.patchdiff[iv][iii] = max(0.0, self.stencil[iv][iii])
        return

        ncells = self.mesh.ncells
        elem = self.mesh.triangles
        for ic in range(ncells):
            for ii in range(3):
                i = elem[ic, ii]
                for jj in range(3):
                    j = elem[ic, jj]
                    for iii, patch in self.mesh.patchinfo[i].items():
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
                        for iii, patch in self.mesh.patchinfo[i].items():
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
        self.formBoundary(du, u)
        self.formCell(du, u)
        if self.upwind == "sl":
            self.formPatchesDiffusionTwo(du, u, self.patchdiff)
        elif self.upwind == "sc":
            self.formSc(du, u)
        elif self.upwind == "slsym":
            self.formPatchesDiffusionTwoSym(du, u)
        return du
    def matrix(self, u=None):
        AB = self.matrixBoundary()
        AI = self.matrixCell()
        if self.upwind == "sl":
            AI += self.matrixPatchesDiffusionTwo(u)
        elif self.upwind == "sc":
            AI += self.matrixSc(u)
        elif self.upwind == "slsym":
            AI += self.matrixPatchesDiffusionTwoSym(u)
        A = (AI + AB).tocsr()
        # self.checkMMatrix(A)
        return A
    def formSc(self, du, u):
        for iv in range( self.mesh.nnodes):
            for ii, patch in self.mesh.patchinfo[iv].items():
                bndiff = self.sc[iv]
                iv2 = patch[0][0]
                du[iv] += bndiff*(u[iv]-u[iv2])
                du[iv2] -= bndiff * (u[iv] - u[iv2])
        return du
    def matrixSc(self, u):
        nnodes =  self.mesh.nnodes
        nnpatches = self.nnpatches
        nplocal = 4
        index = np.zeros(nplocal * nnpatches, dtype=int)
        jndex = np.zeros(nplocal * nnpatches, dtype=int)
        A = np.zeros(nplocal * nnpatches, dtype=np.float64)
        count = 0
        for iv in range( nnodes):
            bndiff = self.sc[iv]
            for ii, patch in self.mesh.patchinfo[iv].items():
                iv2 = patch[0][0]
                index[count:count+2] = iv
                index[count+2:count+4] = iv2
                jndex[count + 0] = iv
                jndex[count + 1] = iv2
                jndex[count + 2] = iv
                jndex[count + 3] = iv2
                A[count + 0] += bndiff
                A[count + 1] -= bndiff
                A[count + 2] -= bndiff
                A[count + 3] += bndiff
                count += nplocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))

    def formCell(self, du, u):
        for iv in range( self.mesh.nnodes):
            for ii,patch in self.mesh.patchinfo[iv].items():
                jv = patch[0][0]
                du[iv]  += self.stencil[iv][ii]*(u[jv]-u[iv])
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
        # A = np.around(A, 16) + 0
        A = scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
        return A


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

    from transportcg1supgdynamic import TransportCg1SupgDynamic
    methods = {}
    methods['xilin'] = TransportCg1Supg(problem=problem, upwind = "sl", xi="xilin", coef="mh")
    # methods['xiconst'] = TransportCg1Supg(problem=problem, upwind = "sl", xi="xiconst", coef="mh")
    methods['xi'] = TransportCg1Supg(problem=problem, upwind = "sl", xi="xi", coef="mh")
    # methods['xi2'] = TransportCg1Supg(problem=problem, upwind = "sl", xi="xi2", coef="mh")
    methods['xi3'] = TransportCg1Supg(problem=problem, upwind = "sl", xi="xi3", coef="mh")
    methods['supg'] = TransportCg1Supg(problem=problem, upwind = "supg", coef="supg")
    # methods['bizarre'] = TransportCg1Supg(problem=problem, upwind = "bizarre")
    # methods['supgsc'] = TransportCg1Supg(problem=problem, upwind = "sc")
    # methods['xi2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi="xi")
    # methods['supg2'] = TransportCg1Supg(problem=problem, upwind = "supg", coef="mh")
    # methods['xi'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi')
    # methods['xilin'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xilin', coef="mh")
    # methods['xi2'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi', coef="mh")
    # methods['xib'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xibis')
    # methods['xit'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xiter')
    # methods['xiq'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xiquater')
    # methods['xi2'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi2')
    # methods['xi3'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi3')
    # methods['xisca3'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi3', shockcapturing="phiabs")
    # methods['xiscc3'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi3', shockcapturing="phicon")
    # methods['xisca'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xi', shockcapturing="phiabs")
    # methods['xinew'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xinew')
    # methods['xinew2'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xinew2')
    # methods['xisignmin'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xisignmin')
    # methods['xispline'] = TransportCg1Supg(problem=problem, upwind = "sl", xi='xispline')

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    print("before CompareErrors")
    niter = 4
    h = [1.0*np.power(0.5,i)  for i in range(niter)]
    compareerrors.compare(h=h)