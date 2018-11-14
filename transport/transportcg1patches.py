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
from fem.femp12d import FemP12D
from transportcg1 import TransportCg1
from tools.comparerrors import CompareErrors
from mesh.trimeshwithnodepatches import TriMeshWithNodePatches
import scipy.linalg as linalg


class TransportCg1Patches(TransportCg1):
    """
    Attention :  le schéma par patch n'est pas monotone, car on fait l'approximation 0.5( |a|+|b|) pour le max
    """
    def __init__(self, **kwargs):
        self.upwind = kwargs.pop('upwind')
        if kwargs.has_key('diff'):
            self.diff = kwargs.pop('diff')
        else:
            self.diff = 'min'
        if kwargs.has_key('timescheme'):
            self.timescheme = kwargs.pop('timescheme')
        TransportCg1.__init__(self, **kwargs)
        if self.finaltime is not None:
            self.nvisusteps = 111
            self.cfl = 1./12.

    def setMesh(self, mesh):
        TransportCg1.setMesh(self, mesh)
        self.mesh = TriMeshWithNodePatches(mesh)
        if self.upwind=="two" or self.upwind=="all":
            self.computepatchdiff()
        if self.finaltime is not None:
            if self.timescheme=="explicit":
                self.M = self.massMatrixLumped()
            else:
                self.M = self.massMatrixLumped()

    def computepatchdiff(self, u=None):
        for iv in range(self.mesh.nnodes):
            n = self.mesh.patchinfo[iv][0][1].shape[0]
            fac = 1.0
            if self.upwind=="all": fac = float(n)
            for ii, patch in enumerate(self.mesh.patches_bnodes[iv]):
                if patch[5] == -1: continue
                bn = (1.0 / 6.0) * (patch[3] * self.betan[patch[1]] + patch[4] * self.betan[patch[2]])
                self.patchdiff[iv][ii] = fac*max(bn,0.0)
        for i in range(self.mesh.nnodes):
            sc = 1
            boundary = -1 in  self.mesh.patches_bnodes[i][:, 5]
            if u is not None and not boundary:
                g, g2 = self.computePatchGradients(i, u)
                sc = self.phi(g, g2)
            # print sc
            # print self.patchdiff[i]
            self.patchdiff[i] = np.around(sc*self.patchdiff[i], 16) + 0

    def computeRhsDynamic(self, uold):
        self.b[:]= 0.0
        self.b = - self.formBoundary(self.b, uold)
        if self.timescheme == "explicit":
            sl = True
            if sl:
                for iv in range(self.mesh.nnodes):
                    boundary = -1 in self.mesh.patches_bnodes[iv][:, 5]
                    for ii, patch in self.mesh.patchinfo[iv].iteritems():
                        patch2 = self.mesh.patches_bnodes[iv][ii]
                        iv2 = patch[0][0]
                        assert iv2==patch2[0]
                        bn = (1.0 / 6.0) * (patch2[3] * self.betan[patch2[1]] + patch2[4] * self.betan[patch2[2]])
                        gradu = uold[iv2] - uold[iv]
                        # gradu2=0.0
                        gradu2 = self.computeCorrectionPatchTwo(boundary, iv, iv2, patch, gradu, uold)
                        bndiff = max(0.0, bn)
                        # bndiff = abs(bn)
                        self.b[iv] -= bn * uold[iv2]
                        self.b[iv] += bndiff * (gradu - gradu2)
                        self.b[iv2] -= bndiff * (gradu - gradu2)
            else:
                for iv in range(self.mesh.nnodes):
                    for ii, patch in enumerate(self.mesh.patches_bnodes[iv]):
                        iv2 = patch[0]
                        bn = (1.0 / 6.0) * (patch[3] * self.betan[patch[1]] + patch[4] * self.betan[patch[2]])
                        self.b[iv] -= bn * uold[iv2]
            ncells =  self.mesh.ncells
            elem =  self.mesh.triangles
            for ic in range(ncells):
                for ii in range(3):
                    self.b[elem[ic, ii]] += self.mesh.area[ic]/3.0/self.dt*uold[elem[ic, ii]]
        else:
            return self.b
            raise ValueError("unknwon timescheme '%s'" %timescheme )
        self.b = np.around(self.b, 16) + 0
        self.computepatchdiff(uold)
        return self.b
    def massMatrixLumped(self):
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        index = np.zeros(3 * ncells, dtype=int)
        A = np.zeros(3 * ncells, dtype=np.float64)
        for ic in range(ncells):
            for ii in range(3):
                index[3*ic + ii] = elem[ic, ii]
                A[3*ic + ii] += self.mesh.area[ic]/3.0/self.dt
        A = np.around(A, 16) + 0
        return scipy.sparse.coo_matrix((A, (index, index)), shape=(nnodes, nnodes)).tocsr()

    def matrixDynamic(self, u):
        if self.timescheme == "explicit":
            return self.M
        # self.A = self.M + self.matrix(u)
        self.A = self.M + 0.5*self.matrix(0.5*(u+self.umoinsun))
        return self.A

    def residualDynamic(self, u):
        if self.timescheme == "explicit":
            # self.r = self.M*u - self.b
            # print 'u-uold', linalg.norm(u-self.umoinsun)
            self.r[:] = 0.0
            self.form(self.r, self.umoinsun)
            self.r += self.M * (u - self.umoinsun)  # - self.b
            # self.r += self.form(self.r, self.umoinsun)
            return np.around(self.r, 16)
        self.r[:] = 0.0
        # self.form(self.r, u)
        # self.r += self.M * (u - self.umoinsun)
        self.form(self.r, 0.5*(u+self.umoinsun))
        self.r += self.M * (u - self.umoinsun)
        return np.around(self.r, 16)

    def form(self, du, u):
        self.formReaction(du, u)
        self.formInteriorPatches(du, u)
        if self.upwind == 'centered':
            pass
        elif self.upwind == 'two':
            self.formPatchesDiffusionTwo(du, u, self.patchdiff)
        elif self.upwind == 'all':
            self.formPatchesDiffusionAll(du, u)
        else:
            raise ValueError("unknown upwind", self.upwind)
        self.formBoundary(du, u)
        return du
    def matrix(self, u=None):
        AR = self.matrixReaction()
        AI = self.matrixInteriorPatches()
        if self.upwind == 'centered':
            pass
        elif self.upwind == 'two':
            AI += self.matrixPatchesDiffusionTwo(u)
        elif self.upwind == 'all':
            AI += self.matrixPatchesDiffusionAll(u)
        else:
            raise ValueError("unknown upwind", self.upwind)
        AB = self.matrixBoundary()
        A = (AI + AB + AR).tocsr()
        # tools.sparse.checkMmatrix(A)
        return A

    def formInteriorPatches(self, du, u):
        for iv in range( self.mesh.nnodes):
            for patch in  self.mesh.patches_bnodes[iv]:
                iv2 = patch[0]
                bn = (1.0 / 6.0) *( patch[3] * self.betan[patch[1]] + patch[4] * self.betan[patch[2]])
                du[iv] +=  bn * u[iv2]
        return du
    def matrixInteriorPatches(self):
        nnodes =  self.mesh.nnodes
        index = np.zeros(self.nnpatches, dtype=int)
        jndex = np.zeros(self.nnpatches, dtype=int)
        A = np.zeros(self.nnpatches, dtype=np.float64)
        count = 0
        for iv in range( nnodes):
            for patch in  self.mesh.patches_bnodes[iv]:
                index[count] = iv
                jndex[count] = patch[0]
                bn = patch[3] * self.betan[patch[1]] + patch[4] * self.betan[patch[2]]
                A[count] += (1.0 / 6.0) * bn
                count += 1
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))


    # def solveNonlinear(self, u=None):
    #     return TransportCg1.solveNonlinear(self, u, rtol=1e-15, gtol=1e-16, maxiter=5, checkmaxiter=False)

# ------------------------------------- #

if __name__ == '__main__':
    # problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    # problem = 'Analytic_Exponential'
    # problem = 'Analytic_Linear'
    # problem = 'RotStat'
    # problem = 'Ramp'
    alpha = 1.0
    alpha = 0.0
    # beta = lambda x, y: (-y, x)
    beta = lambda x, y: (-np.cos(np.pi * (x + y)), np.cos(np.pi * (x + y)))
    beta = lambda x, y: (-np.sin(np.pi * x) * np.cos(np.pi * y), np.sin(np.pi * y) * np.cos(np.pi * x))
    beta = lambda x, y: (-np.cos(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * y) * np.sin(np.pi * x))
    beta = None
    #
    methods = {}
    methods['centered'] = TransportCg1Patches(upwind='centered', problem=problem, alpha=alpha, beta=beta)
    # methods['twoxi'] = TransportCg1Patches(upwind='two', xi='xi', problem=problem, alpha=alpha, beta=beta)
    # methods['allxi'] = TransportCg1Patches(upwind='all', xi='xi', problem=problem, alpha=alpha, beta=beta)
    methods['two'] = TransportCg1Patches(upwind='two', problem=problem, alpha=alpha, beta=beta)
    methods['all'] = TransportCg1Patches(upwind='all', problem=problem, alpha=alpha, beta=beta)

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    niter = 5
    h = [0.4*np.power(0.5,i)  for i in range(niter)]
    compareerrors.compare(h=h)
