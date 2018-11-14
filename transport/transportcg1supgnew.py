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
from mesh.trimeshwithnodepatches import TriMeshWithNodePatches
from mesh.trimeshwithnodepatches import TriMeshWithNodePatches


class TransportCg1SupgNew(TransportCg1):
    """
    """
    def __init__(self, **kwargs):
        if kwargs.has_key('upwind'):
            self.upwind = kwargs.pop('upwind')
        else:
            self.upwind = "supg"
        print self.upwind
        TransportCg1.__init__(self, **kwargs)
        # self.fem = FemP12D(self.mesh)

    def setMesh(self, mesh):
        TransportCg1.setMesh(self, mesh)
        self.mesh = TriMeshWithNodePatches(mesh)
        self.fem.setMesh(mesh)
        beta = np.stack( self.beta(self.mesh.centersx, self.mesh.centersy), axis=-1 )
        self.downwindinfo = self.mesh.computeDownwindInfo(beta)
        # self.mesh.plotDownwindInfo(self.downwindinfo)
        # sys.exit(1)
        if self.upwind == "side":
            self.downwindinfoedge = self.computeDownwindInfoEdges(self.downwindinfo)
        elif self.upwind == "patch":
            self.computePatchDiff(self.downwindinfo)

    def computePatchDiff(self, downwindinfo):
        ncells = self.mesh.ncells
        for ic in range(ncells):
            i0 = downwindinfo[0][ic, 0]
            i1 = downwindinfo[0][ic, 1]
            delta = downwindinfo[1][ic, 0]
            p = downwindinfo[1][ic, 1]
            fq0 = p*max(0,2.0/3.0-p) * self.mesh.area[ic] / delta
            fq1 = (1-p)*max(0, p-1./3.) * self.mesh.area[ic] / delta
            i2=-1
            for ii in range(3):
                i2c = self.mesh.triangles[ic,ii]
                if i2c != i0 and i2c !=i1:
                    i2 = i2c
                    break
            for ii, patch in self.mesh.patchinfo[i0].iteritems():
                # self.patchdiff[i0][ii] = fq0 * patch[1].shape[0]
                if patch[0][0] == i1:
                    self.patchdiff[i0][ii] = fq0
            for ii, patch in self.mesh.patchinfo[i1].iteritems():
                # self.patchdiff[i1][ii] = fq1 * patch[1].shape[0]
                if patch[0][0] == i0:
                    self.patchdiff[i1][ii] = fq1
    def computeDownwindInfoEdges(self, downwindinfo):
        ncells = self.mesh.ncells
        patchinfoindices = -1*np.ones( (ncells, 2, 2) , dtype=np.int64)
        patchinfocoefs = np.zeros( (ncells, 2, 3) , dtype=np.float64)
        for ic in range(ncells):
            i0 = downwindinfo[0][ic, 0]
            i1 = downwindinfo[0][ic, 1]
            delta = downwindinfo[1][ic, 0]
            p = downwindinfo[1][ic, 1]
            fq0 = p*max(0,2.0/3.0-p)
            fq1 = (1-p)*max(0, p-1./3.)
            patchinfocoefs[ic, 0, 0] = fq0 * self.mesh.area[ic] / delta
            patchinfocoefs[ic, 1, 0] = fq1 * self.mesh.area[ic] / delta
            # if fq0 >= fq1:
            #     patchinfocoefs[ic, 0, 0] = fq0*self.mesh.area[ic]/delta
            #     patchinfocoefs[ic, 1, 0] = 0.0
            # else:
            #     patchinfocoefs[ic, 0, 0] = 0.0
            #     patchinfocoefs[ic, 1, 0] = fq1*self.area[ic]/delta
            for jj in range(2):
                icenter = downwindinfo[0][ic, jj]
                iother = downwindinfo[0][ic, 1-jj]
                for ii, patch in self.mesh.patchinfo[icenter].iteritems():
                    if patch[0][0]==iother:
                        index = patch[0][2]
                        if index == -1: continue
                        patchinfoindices[ic, jj, 0] = patch[1][index][0]
                        patchinfoindices[ic, jj, 1] = patch[1][index][1]
                        patchinfocoefs[ic, jj, 1] = patch[2][index][0]
                        patchinfocoefs[ic, jj, 2] = patch[2][index][1]
                        break
        return [patchinfoindices, patchinfocoefs]


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
                i0 = self.downwindinfo[0][ic, 0]
                i1 = self.downwindinfo[0][ic, 1]
                delta = self.downwindinfo[1][ic, 0]
                p = self.downwindinfo[1][ic, 1]
                b[i0] += p*bcells[ic]
                b[i1] += (1-p)*bcells[ic]
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
        self.formReaction(du, u)
        self.formCell(du, u)
        if self.upwind == "side":
            self.formSide(du, u, nl=True)
        elif self.upwind == "patch":
            # self.formPatchesDiffusionAll(du, u,nl=True)
            self.formPatchesDiffusionTwo(du, u, nl=True)
        self.formBoundary(du, u)
        return du
    def matrix(self, u=None):
        AR = self.matrixReaction(u)
        AI = self.matrixCell()
        if self.upwind == "side":
            AI += self.matrixSide(u, nl=True)
        elif self.upwind == "patch":
            # AI += self.matrixPatchesDiffusionAll(u, nl=True)
            AI += self.matrixPatchesDiffusionTwo(u, nl=True)
        AB = self.matrixBoundary()
        # self.checkMMatrix(AB.tocsr())
        A = (AI + AB + AR).tocsr()
        # self.checkMMatrix(A)
        return A
    def formCell(self, du, u):
        ncells =  self.mesh.ncells
        elem =  self.mesh.triangles
        for ic in range(ncells):
            i0 = self.downwindinfo[0][ic, 0]
            i1 = self.downwindinfo[0][ic, 1]
            delta = self.downwindinfo[1][ic, 0]
            p = self.downwindinfo[1][ic, 1]
            ubeta = (p*u[i0] + (1-p)*u[i1] - np.mean(u[elem[ic]]))/delta
            du[i0] += self.mesh.area[ic] * p*ubeta
            du[i1] += self.mesh.area[ic] * (1-p)*ubeta
        return du
    def formSide(self, du, u, nl=False):
        ncells = self.mesh.ncells
        for ic in range(ncells):
            pindices, pcoefs = self.downwindinfoedge[0][ic], self.downwindinfoedge[1][ic]
            for jj in range(2):
                icenter = self.downwindinfo[0][ic, jj]
                iother = self.downwindinfo[0][ic, 1-jj]
                boundary = -1 in self.mesh.patches_bnodes[icenter][:, 5]
                ug = (u[iother] - u[icenter])
                w = pcoefs[jj, 0]
                if w==0.0: continue
                if nl:
                    boundary = -1 in self.mesh.patches_bnodes[icenter][:, 5]
                    if boundary:
                        if self.bdrycor:
                            xiv, yiv = self.mesh.x[icenter], self.mesh.y[icenter]
                            xiv2, yiv2 = self.mesh.x[iother], self.mesh.y[iother]
                            ug2 =  self.dirichlet(xiv2, yiv2) - self.dirichlet(xiv, yiv)
                            ug -= ug2
                    else:
                        assert np.all(pindices[jj] > -1)
                        a0 = pcoefs[jj, 1]
                        a1 = pcoefs[jj, 2]
                        ic0 = pindices[jj, 0]
                        ic1 = pindices[jj, 1]
                        ug2 =  a0* (u[ic0] - u[icenter]) + a1 * (u[ic1] - u[icenter])
                        ug2 = self.nl(ug, ug2)
                        ug -= ug2
                du[icenter] -= w * ug
                du[iother] += w * ug
        return du
    def formReaction(self, du, u):
        ncells =  self.mesh.ncells
        elem =  self.mesh.triangles
        for ic in range(ncells):
            i0 = self.downwindinfo[0][ic, 0]
            i1 = self.downwindinfo[0][ic, 1]
            p = self.downwindinfo[1][ic, 1]
            umean = np.mean(u[elem[ic]])
            du[i0] += self.alpha*self.mesh.area[ic] * p*umean
            du[i1] += self.alpha*self.mesh.area[ic] * (1-p)*umean
        #     if self.upwind=="supg": continue
        # #     stbilisierung
        #     ug = u[i0]-u[i1]
        #     stab = max(p,1-p)/3.0
        #     du[i0] += self.alpha * self.mesh.area[ic] * stab * ug
        #     du[i1] -= self.alpha * self.mesh.area[ic] * stab * ug
        #     for i in elem[ic]:
        #         if i!=i0 and i!=i1:
        #             i2 = i
        #     ug = u[i0] - u[i2]
        #     stab = p / 3.0
        #     du[i0] += self.alpha * self.mesh.area[ic] * stab * ug
        #     du[i2] -= self.alpha * self.mesh.area[ic] * stab * ug
        #     ug = u[i1] - u[i2]
        #     stab = (1-p) / 3.0
        #     du[i1] += self.alpha * self.mesh.area[ic] * stab * ug
        #     du[i2] -= self.alpha * self.mesh.area[ic] * stab * ug
        return du
    def matrixReaction(self, u):
        ncells = self.mesh.ncells
        nnodes = self.mesh.nnodes
        elem = self.mesh.triangles
        index = np.zeros(9 * ncells, dtype=int)
        jndex = np.zeros(9 * ncells, dtype=int)
        A = np.zeros(9 * ncells, dtype=np.float64)
        for ic in range(ncells):
            i0 = self.downwindinfo[0][ic, 0]
            i1 = self.downwindinfo[0][ic, 1]
            ii0 = self.downwindinfo[0][ic, 2]
            ii1 = self.downwindinfo[0][ic, 3]
            p = self.downwindinfo[1][ic, 1]
            a0 = self.alpha * self.mesh.area[ic] * p / 3.0
            a1 = self.alpha * self.mesh.area[ic] * (1 - p) / 3.0
            for jj in range(3):
                for ii in range(3):
                    index[9 * ic + ii + 3 * jj] = elem[ic, ii]
                    jndex[9 * ic + ii + 3 * jj] = elem[ic, jj]
            for jj in range(3):
                A[9 * ic + ii0 + 3 * jj] += a0
                A[9 * ic + ii1 + 3 * jj] += a1
            # if self.upwind=="supg": continue
            # #     stabilisierung
            # stab = self.alpha * self.mesh.area[ic] * max(p, 1 - p) / 3.0
            # A[9 * ic + ii0 + 3 * ii0] += stab
            # A[9 * ic + ii0 + 3 * ii1] -= stab
            # A[9 * ic + ii1 + 3 * ii0] -= stab
            # A[9 * ic + ii1 + 3 * ii1] += stab
            # for ii,i in enumerate(elem[ic]):
            #     if i!=i0 and i!=i1:
            #         i2 = i
            #         ii2 = ii
            # A[9 * ic + ii0 + 3 * ii0] += a0
            # A[9 * ic + ii0 + 3 * ii2] -= a0
            # A[9 * ic + ii2 + 3 * ii0] -= a0
            # A[9 * ic + ii2 + 3 * ii2] += a0
            # A[9 * ic + ii1 + 3 * ii1] += a1
            # A[9 * ic + ii1 + 3 * ii2] -= a1
            # A[9 * ic + ii2 + 3 * ii1] -= a1
            # A[9 * ic + ii2 + 3 * ii2] += a1
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))

    def matrixCell(self):
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        index = np.zeros(6 * ncells, dtype=int)
        jndex = np.zeros(6 * ncells, dtype=int)
        A = np.zeros(6 * ncells, dtype=np.float64)
        count=0
        for ic in range(ncells):
            i0 = self.downwindinfo[0][ic, 0]
            i1 = self.downwindinfo[0][ic, 1]
            ii0 = self.downwindinfo[0][ic, 2]
            ii1 = self.downwindinfo[0][ic, 3]
            delta = self.downwindinfo[1][ic, 0]
            p = self.downwindinfo[1][ic, 1]
            for ii in range(3):
                index[count + ii] = i0
                index[count + 3 + ii] = i1
                jndex[count + ii] = elem[ic, ii]
                jndex[count + 3 + ii] = elem[ic, ii]
                A[count + ii] -= self.mesh.area[ic] * p/delta/3.0
                A[count + 3 + ii] -= self.mesh.area[ic] * (1-p) / delta / 3.0
            A[count + ii0] += self.mesh.area[ic] * p*p/delta
            A[count + ii1] += self.mesh.area[ic] * p*(1-p)/delta
            A[count + 3 + ii0] += self.mesh.area[ic] * (1-p)*p/delta
            A[count + 3 + ii1] += self.mesh.area[ic] * (1-p)*(1-p)/delta
            count += 6
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def matrixSide(self, u, nl=False):
        ncells = self.mesh.ncells
        nnodes =  self.mesh.nnodes
        nlocal = 8
        if nl: nlocal += 8
        nlocal2 = nlocal/2
        index = np.zeros(nlocal * ncells, dtype=int)
        jndex = np.zeros(nlocal * ncells, dtype=int)
        A = np.zeros(nlocal * ncells, dtype=np.float64)
        count = 0
        for ic in range(ncells):
            pindices, pcoefs = self.downwindinfoedge[0][ic], self.downwindinfoedge[1][ic]
            # jj = np.argmax(pcoefs[:,0])
            for jj in range(2):
                w = pcoefs[jj,0]
                icenter = self.downwindinfo[0][ic, jj]
                iother = self.downwindinfo[0][ic, 1-jj]
                boundary = -1 in self.mesh.patches_bnodes[icenter][:, 5]
                ug = (u[iother] - u[icenter])
                index[count + 0+ nlocal2*jj] = icenter
                index[count + 1+ nlocal2*jj] = icenter
                index[count + 2+ nlocal2*jj] = iother
                index[count + 3+ nlocal2*jj] = iother
                jndex[count + 0+ nlocal2*jj] = icenter
                jndex[count + 1+ nlocal2*jj] = iother
                jndex[count + 2+ nlocal2*jj] = icenter
                jndex[count + 3+ nlocal2*jj] = iother
                A[count + 0+ nlocal2*jj] += w
                A[count + 1+ nlocal2*jj] -= w
                A[count + 2+ nlocal2*jj] -= w
                A[count + 3+ nlocal2*jj] += w
                if nl:
                    boundary = -1 in self.mesh.patches_bnodes[icenter][:, 5]
                    if not boundary:
                        assert np.all(pindices[jj] > -1)
                        ug = (u[iother] - u[icenter])
                        a0 = pcoefs[jj, 1]
                        a1 = pcoefs[jj, 2]
                        ic0 = pindices[jj, 0]
                        ic1 = pindices[jj, 1]
                        ug2 = a0 * (u[ic0] - u[icenter]) + a1 * (u[ic1] - u[icenter])
                        dx = self.dnldx(ug, ug2)
                        dy = self.dnldy(ug, ug2)
                        index[count + 4+ nlocal2*jj] = icenter
                        index[count + 5+ nlocal2*jj] = icenter
                        index[count + 6+ nlocal2*jj] = iother
                        index[count + 7+ nlocal2*jj] = iother
                        jndex[count + 4+ nlocal2*jj] = ic0
                        jndex[count + 5+ nlocal2*jj] = ic1
                        jndex[count + 6+ nlocal2*jj] = ic0
                        jndex[count + 7+ nlocal2*jj] = ic1
                        A[count + 4+ nlocal2*jj] += w*dy * a0
                        A[count + 5+ nlocal2*jj] += w*dy * a1
                        A[count + 6+ nlocal2*jj] -= w*dy * a0
                        A[count + 7+ nlocal2*jj] -= w*dy * a1
                        A[count + 0+ nlocal2*jj] -= w*dx
                        A[count + 1+ nlocal2*jj] += w*dx
                        A[count + 2+ nlocal2*jj] += w*dx
                        A[count + 3+ nlocal2*jj] -= w*dx
                        A[count + 0+ nlocal2*jj] -= w*dy * (a0 + a1)
                        A[count + 2+ nlocal2*jj] += w*dy * (a0 + a1)
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))


# ------------------------------------- #

if __name__ == '__main__':
    problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    # problem = 'Analytic_Exponential'
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Constant'
    # problem = 'RotStat'
    problem = 'Ramp'
    alpha = 100.0
    alpha = 0.0
    # beta = lambda x, y: (-y, x)
    beta = lambda x, y: (-np.cos(np.pi * (x + y)), np.cos(np.pi * (x + y)))
    beta = lambda x, y: (-np.sin(np.pi * x) * np.cos(np.pi * y), np.sin(np.pi * y) * np.cos(np.pi * x))
    beta = lambda x, y: (-np.cos(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * y) * np.sin(np.pi * x))
    beta = None

    methods = {}
    # methods['lin'] = TransportCg1SupgNew(problem=problem, alpha=alpha, beta=beta)
    methods['supg'] = TransportCg1SupgNew(upwind="supg",problem=problem, alpha=alpha, beta=beta)
    methods['side'] = TransportCg1SupgNew(upwind="side", xi='xisignmin', problem=problem, alpha=alpha, beta=beta)
    methods['patch'] = TransportCg1SupgNew(upwind="patch", xi='xisignmin', problem=problem, alpha=alpha, beta=beta)

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    h = [1.0, 0.5, 0.25, 0.125]
    # h = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03]
    compareerrors.compare(h=h)
