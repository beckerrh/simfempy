# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse
import xifunction

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transport import Transport
import phifunction
import xifunction

class TransportCg1(Transport):
    """
    cense etre derivee
    """
    def __init__(self, **kwargs):
        if 'sym' in kwargs:
            self.sym = kwargs.pop('sym')
        else:
            self.sym = False
        if 'xi' in kwargs:
            self.xi = kwargs.pop('xi')
        else:
            self.xi = 'xisignmin'
        if 'shockcapturing' in kwargs:
            self.phi = kwargs.pop('shockcapturing')
        else:
            self.phi = 'phione'
        Transport.__init__(self, **kwargs)
        if self.mesh:
            self.setMesh(self.mesh)
        self.constructLimiter(self.xi, self.phi)
        try:
            self.bdrycor =  kwargs['problem'].find('Analytic') >= 0
        except:
            self.bdrycor = False
        self.patchdiff = None
    def constructLimiter(self, xi, phi):
        print('xi', xi, 'phi', phi)
        if phi=='phione':
            self.phi = phifunction.phione
            self.dphi = phifunction.dphione
        elif phi == 'phiabs':
            self.phi = phifunction.phiabs
            self.dphi = phifunction.dphiabs
        elif phi == 'phisum':
            self.phi = phifunction.phisum
            self.dphi = phifunction.dphisum
        elif phi == 'phicon':
            self.phi = phifunction.phicon
            self.dphi = phifunction.dphicon
        else:
            raise ValueError("unknown shockcapturing %s" %phi)
        if xi =='xi':
            self.nl = xifunction.xi
            self.dnldx = xifunction.dxidx
            self.dnldy = xifunction.dxidy
        elif xi == 'xi2':
            self.nl = xifunction.xi2
            self.dnldx = xifunction.dxi2dx
            self.dnldy = xifunction.dxi2dy
        elif xi == 'xi3':
            self.nl = xifunction.xi3
            self.dnldx = xifunction.dxi3dx
            self.dnldy = xifunction.dxi3dy
        elif xi == 'xibis':
            self.nl = xifunction.xibis
            self.dnldx = xifunction.dxibisdx
            self.dnldy = xifunction.dxibisdy
        elif xi == 'xiter':
            self.nl = xifunction.xiter
            self.dnldx = xifunction.dxiterdx
            self.dnldy = xifunction.dxiterdy
        elif xi == 'xiquater':
            self.nl = xifunction.xiquater
            self.dnldx = xifunction.dxiquaterdx
            self.dnldy = xifunction.dxiquaterdy
        elif xi == 'xispline':
            self.nl = xifunction.xispline
            self.dnldx = xifunction.dxisplinedx
            self.dnldy = xifunction.dxisplinedy
        elif xi == 'xinew':
            self.nl = xifunction.xinew
            self.dnldx = xifunction.dxinewdx
            self.dnldy = xifunction.dxinewdy
        elif xi == 'xinew2':
            self.nl = xifunction.xinew2
            self.dnldx = xifunction.dxinew2dx
            self.dnldy = xifunction.dxinew2dy
        elif xi=='xisignmin':
            self.nl = xifunction.xisignmin
            self.dnldx = xifunction.dxisignmindx
            self.dnldy = xifunction.dxisignmindy
        elif xi=='xilin':
            self.nl = xifunction.xilin
            self.dnldx = xifunction.dxilindx
            self.dnldy = xifunction.dxilindy
        elif xi == 'xiconst':
            self.nl = xifunction.xiconst
            self.dnldx = xifunction.dxiconstdx
            self.dnldy = xifunction.dxiconstdy
        else:
            raise ValueError('unknown limiter %s' %xi)
    def setMesh(self, mesh):
        Transport.setMesh(self, mesh)
        # self.mesh = TriMeshWithNodePatches(mesh)
        self.mesh.computeNodePatches()
        self.nnpatches = 0
        self.npatchmax = 0
        self.npatchmax2 = 0
        self.patchdiff = {}
        self.sc = np.ones(self.mesh.nnodes, dtype=np.float64)
        for iv in range( self.mesh.nnodes):
            npatch2 = self.mesh.patchinfo[iv][0][1].shape[0]
            self.npatchmax2 = max(self.npatchmax2, npatch2)
            self.npatchmax = max(self.npatchmax, len(self.mesh.patchinfo[iv]))
            npatch = len(self.mesh.patchinfo[iv])
            self.patchdiff[iv] = np.zeros(npatch, dtype=np.float64)
            for ii, patch in self.mesh.patchinfo[iv].items():
                npatch = patch[1].shape[0]
                self.nnpatches += npatch

    def computeShockCapturing(self, u):
        for iv in range( self.mesh.nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            if boundary:
                self.sc[iv] = 1.0
            else:
                g, g2 = self.computePatchGradients(iv, u)
                self.sc[iv] = self.phi(g, g2)

    def computeEdgeDiffusion(self):
        nedges = self.mesh.nedges
        self.edgebeta = np.zeros((nedges,2), dtype=np.float64)
        for iv in range(self.mesh.nnodes):
            for i, patch in enumerate(self.mesh.patches_bnodes[iv]):
                if patch[5] == -1: continue
                bn = (1.0 / 6.0) * (patch[3] * self.betan[patch[1]] + patch[4] * self.betan[patch[2]])
                ie = patch[5]
                iv2 = patch[0]
                if self.mesh.edges[ie,0] == iv:
                    assert self.mesh.edges[ie,1] == iv2
                    self.edgebeta[ie,0] += bn
                else:
                    assert self.mesh.edges[ie, 0] == iv2
                    assert self.mesh.edges[ie, 1] == iv
                    self.edgebeta[ie,1] += bn
        if self.upwind == "centered":
            return
        self.edgediff = np.zeros(nedges, dtype=np.float64)
        if self.diff == 'min':
            for ie in range(nedges):
                self.edgediff[ie] = max(max(self.edgebeta[ie,0], 0.0),max(self.edgebeta[ie,1],0.0))
        else:
            for ie in range(nedges):
                if -1 in self.mesh.cellsOfEdge[ie]:
                    for ii in range(2):
                        it = self.mesh.cellsOfEdge[ie, ii]
                        if it == -1: continue
                        be = self.betan[ie]
                        if ii == 1:
                            be *= -1
                        for iii in range(3):
                            ie2 = self.mesh.edgesOfCell[it, iii]
                            if ie2 == ie:
                                continue
                            else:
                                b2 = self.betan[ie2]
                                if self.mesh.cellsOfEdge[ie2, 1] == it:
                                    b2 *= -1
                                b2pbe = b2 + be
                                if b2pbe < 0.0: b2pbe = 0.0
                                if b2 > 0.0: b2 = 0.0
                                self.edgediff[ie] = (1.0 / 6.0) * max(b2pbe, -b2)
                                break
                    continue
                for ii in range(2):
                    it = self.mesh.cellsOfEdge[ie, ii]
                    assert it != -1
                    found = 0
                    bn = 0.0
                    for iii in range(3):
                        ie2 = self.mesh.edgesOfCell[it, iii]
                        if ie2 == ie: continue
                        if found == 0:
                            found += 1
                            if self.mesh.cellsOfEdge[ie2, 0] == it:
                                bn += self.betan[ie2]
                            else:
                                bn -= self.betan[ie2]
                        else:
                            found += 1
                            if self.mesh.cellsOfEdge[ie2, 0] == it:
                                bn -= self.betan[ie2]
                            else:
                                bn += self.betan[ie2]
                    if found != 2:
                        raise ValueError('found=%d' % (found))
                    bn = (1.0 / 6.0) * np.abs(bn)
                    if bn > self.edgediff[ie]:
                        self.edgediff[ie] = bn
        self.edgediff = np.around(self.edgediff, 14) + 0
    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        ncells =  self.mesh.ncells
        nnodes =  self.mesh.nnodes
        elem =  self.mesh.triangles
        bdryedges =  self.mesh.bdryedges
        bcells = self.computeBcells(rhs)
        b = np.zeros(nnodes)
        if bcells is not None:
            for ic in range(ncells):
                for ii in range(3):
                    b[elem[ic, ii]] += bcells[ic] / 3.0
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
    def formReaction(self, du, u):
      ncells =  self.mesh.ncells
      elem =  self.mesh.triangles
      for ic in range(ncells):
          for ii in range(3):
            du[elem[ic, ii]] += u[elem[ic, ii]]*self.alpha* self.mesh.area[ic] / 3.0
      return du
    def formBoundary(self, du, u):
        nnodes =  self.mesh.nnodes
        bdryedges =  self.mesh.bdryedges
        nbdryedges = len(bdryedges)
        for (count, ie) in enumerate(bdryedges):
            bn = self.betan[ie]
            iv0 =  self.mesh.edges[ie, 0]
            iv1 =  self.mesh.edges[ie, 1]
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie, 0]
                if ic > -1:
                    du[iv0] -= 0.5 * bn * u[iv0]
                    du[iv1] -= 0.5 * bn * u[iv1]
            else:
                ic =  self.mesh.cellsOfEdge[ie, 1]
                if ic > -1:
                    du[iv0] += 0.5 * bn * u[iv0]
                    du[iv1] += 0.5 * bn * u[iv1]
        return du
    def matrixReaction(self):
      ncells =  self.mesh.ncells
      nnodes =  self.mesh.nnodes
      elem =  self.mesh.triangles
      index = np.zeros(nnodes, dtype=int)
      jndex = np.zeros(nnodes, dtype=int)
      A = np.zeros(nnodes, dtype=np.float64)
      for iv in range(nnodes):
        index[iv] = iv
        jndex[iv] = iv
      for ic in range(ncells):
          for ii in range(3):
            A[elem[ic, ii]] += self.alpha* self.mesh.area[ic] / 3.0
      return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def matrixBoundary(self):
        nnodes =  self.mesh.nnodes
        nbdryedges = len( self.mesh.bdryedges)
        bdryedges =  self.mesh.bdryedges
        index = np.zeros(2 * nbdryedges, dtype=int)
        jndex = np.zeros(2 * nbdryedges, dtype=int)
        A = np.zeros(2 * nbdryedges, dtype=np.float64)
        for (count, ie) in enumerate(bdryedges):
            bn = self.betan[ie]
            iv0 =  self.mesh.edges[ie, 0]
            iv1 =  self.mesh.edges[ie, 1]
            index[2 * count] = iv0
            jndex[2 * count] = iv0
            index[2 * count + 1] = iv1
            jndex[2 * count + 1] = iv1
            if bn < 0.0:
                ic =  self.mesh.cellsOfEdge[ie, 0]
                if ic > -1:
                    # print 'inflow edge (-)', ie, 'cell', ic, bn
                    A[2 * count] -= 0.5 * bn
                    A[2 * count + 1] -= 0.5 * bn
            else:
                ic =  self.mesh.cellsOfEdge[ie, 1]
                if ic > -1:
                    # print 'inflow edge (+)', ie, 'cell', ic, bn
                    A[2 * count] += 0.5 * bn
                    A[2 * count + 1] += 0.5 * bn
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def computeInitialcondition(self):
        nnodes = self.mesh.nnodes
        u = np.zeros(nnodes)
        for i in range(nnodes):
            u[i] = self.initialcondition(self.mesh.x[i], self.mesh.y[i])
        return u

    def computeCorrectionPatchAll(self, isbdry, iv, iv2, patch, gradu, u):
        if isbdry:
            if self.bdrycor:
                xiv, yiv = self.mesh.x[iv], self.mesh.y[iv]
                xiv2, yiv2 = self.mesh.x[iv2], self.mesh.y[iv2]
                return self.dirichlet(xiv2, yiv2) - self.dirichlet(xiv, yiv)
            return 0.0
        n = patch[1].shape[0]
        grad2 = np.zeros(n)
        for ii in range(n):
            a0 = patch[2][ii][0]
            a1 = patch[2][ii][1]
            grad2[ii] = a0 * (u[patch[1][ii][0]] - u[iv]) +a1 * (u[patch[1][ii][1]] - u[iv])
            grad2[ii] = self.nl(gradu, grad2[ii])
        return np.mean(grad2)
    def formPatchesDiffusionAll(self, du, u, nl=False):
        for iv in range( self.mesh.nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            for ii,patch in self.mesh.patchinfo[iv].items():
                bndiff = self.patchdiff[iv][ii]
                iv2 = patch[0][0]
                gradu = u[iv2] - u[iv]
                gradu2 = 0.0
                if nl:
                    gradu2 = self.computeCorrectionPatchAll(boundary, iv, iv2, patch, gradu, u)
                du[iv] -= bndiff * (gradu-gradu2)
                du[iv2] += bndiff * (gradu-gradu2)
        return du
    def computeCorrectionPatchDerAll(self, isbdry, iv, iv2, patch, u):
        n = patch[1].shape[0]
        coef = np.zeros(4+4*n, dtype=np.float64)
        ind = np.zeros(4*n, dtype=int)
        jnd = np.zeros(4*n, dtype=int)
        if isbdry:
            return ind, jnd, coef
        gradu = u[iv2] - u[iv]
        for ii in range(n):
            ii0 = patch[1][ii][0]
            ii1 = patch[1][ii][1]
            a0 = patch[2][ii][0]
            a1 = patch[2][ii][1]
            ind[2*ii + 0] = iv
            ind[2*ii + 1] = iv
            ind[2*ii + 2*n + 0] = iv2
            ind[2*ii + 2*n + 1] = iv2
            jnd[2*ii + 0] = ii0
            jnd[2*ii + 1] = ii1
            jnd[2*ii + 2*n + 0] = ii0
            jnd[2*ii + 2*n + 1] = ii1
            grad2 = a0 * (u[ii0] - u[iv]) +a1 * (u[ii1] - u[iv])
            dx = self.dnldx(gradu, grad2)/n
            dy = self.dnldy(gradu, grad2)/n
            coef[4+2*ii+0] += dy * a0
            coef[4+2*ii+1] += dy * a1
            coef[1] -= dy * (a0 + a1)
            coef[4+2*ii+0+2*n] -= dy * a0
            coef[4+2*ii+1+2*n] -= dy * a1
            coef[3] += dy * (a0 + a1)
            coef[0] += dx
            coef[1] -= dx
            coef[2] -= dx
            coef[3] += dx
        return ind, jnd, coef
    def matrixPatchesDiffusionAll(self, u=None, nl=False):
        nnodes =  self.mesh.nnodes
        nnpatches = self.nnpatches
        nplocal = 4
        if nl:
            nplocal += 4*self.npatchmax2
        index = np.zeros(nplocal * nnpatches, dtype=int)
        jndex = np.zeros(nplocal * nnpatches, dtype=int)
        A = np.zeros(nplocal * nnpatches, dtype=np.float64)
        count = 0
        # countbdry=0
        for iv in range( nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            for ii, patch in self.mesh.patchinfo[iv].items():
                bndiff = self.patchdiff[iv][ii]
                iv2 = patch[0][0]
                # if iv2 == iv:
                #     countbdry += 1
                #     continue
                index[count + 0] = iv
                index[count + 1] = iv
                index[count + 2] = iv2
                index[count + 3] = iv2
                jndex[count + 0] = iv2
                jndex[count + 1] = iv
                jndex[count + 2] = iv2
                jndex[count + 3] = iv
                A[count + 0] -= bndiff
                A[count + 1] += bndiff
                A[count + 2] += bndiff
                A[count + 3] -= bndiff
                if nl:
                    ind, jnd, coef = self.computeCorrectionPatchDerAll(boundary, iv, iv2, patch, u)
                    n = ind.shape[0]
                    index[count+4:count+4+n] = ind[:]
                    jndex[count+4:count+4+n] = jnd[:]
                    A[count:count+4+n] += bndiff*coef[:]
                count += nplocal
            # assert countbdry==self.mesh.nbdryvert
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))

    def formPatchesDiffusionTwoLinear(self, du, u):
        for iv in range( self.mesh.nnodes):
            for ii, patch in self.mesh.patchinfo[iv].items():
                bndiff = self.patchdiff[iv][ii]
                iv2 = patch[0][0]
                du[iv] += bndiff*(u[iv]-u[iv2])
                du[iv2] -= bndiff * (u[iv] - u[iv2])
        return du
    def matrixPatchesDiffusionLinear(self):
        nnodes =  self.mesh.nnodes
        nnpatches = self.nnpatches
        nplocal = 4
        index = np.zeros(nplocal * nnpatches, dtype=int)
        jndex = np.zeros(nplocal * nnpatches, dtype=int)
        A = np.zeros(nplocal * nnpatches, dtype=np.float64)
        count = 0
        for iv in range( nnodes):
            for ii, patch in self.mesh.patchinfo[iv].items():
                bndiff = self.patchdiff[iv][ii]
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

    def computePatchGradients(self, iv, u):
        n = len(self.mesh.patchinfo[iv])
        g = np.zeros(n)
        g2 = np.zeros(n)
        for ii,patch in self.mesh.patchinfo[iv].items():
            iv2 = patch[0][0]
            g[ii] = u[iv2] - u[iv]
            iop = patch[0][2]
            i0 = patch[1][iop][0]
            i1 = patch[1][iop][1]
            a0 = patch[2][iop][0]
            a1 = patch[2][iop][1]
            g2[ii] = a0 * (u[i0] - u[iv]) + a1 * (u[i1] - u[iv])
        return g, g2
    def computeCorrectionPatchTwo(self, isbdry, iv, sc, patchdiff, u):
        n = len(self.mesh.patchinfo[iv])
        phis = np.zeros(n, dtype=np.float64)
        inds = np.zeros(n, dtype=int)
        if isbdry:
            for ii,patch in self.mesh.patchinfo[iv].items():
                bndiff = sc*patchdiff[iv][ii]
                iv2 = patch[0][0]
                inds[ii] = iv2
                gradu = u[iv2] - u[iv]
                gradcor = 0.0
                if self.bdrycor:
                    xiv, yiv = self.mesh.x[iv], self.mesh.y[iv]
                    xiv2, yiv2 = self.mesh.x[iv2], self.mesh.y[iv2]
                    gradcor = self.dirichlet(xiv2, yiv2) - self.dirichlet(xiv, yiv)
                phi = gradu-gradcor
                phis[ii] = bndiff * phi
        else:
            for ii,patch in self.mesh.patchinfo[iv].items():
                bndiff = sc*patchdiff[iv][ii]
                iv2 = patch[0][0]
                inds[ii] = iv2
                gradu = u[iv2] - u[iv]
                iop = patch[0][2]
                i0 = patch[1][iop][0]
                i1 = patch[1][iop][1]
                a0 = patch[2][iop][0]
                a1 = patch[2][iop][1]
                gradu2 = a0 * (u[i0] - u[iv]) +a1 * (u[i1] - u[iv])
                phi = self.nl(gradu, gradu2)
                phis[ii] = bndiff * phi
        return inds, phis
    def formPatchesDiffusionTwo(self, du, u, patchdiff):
        self.computeShockCapturing(u)
        for iv in range( self.mesh.nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            inds, phis = self.computeCorrectionPatchTwo(boundary, iv, self.sc[iv], patchdiff, u)
            n = inds.shape[0]
            for ii in range(n):
                phi = phis[ii]
                iv2 = inds[ii]
                du[iv] -= phi
                du[iv2] += phi
        return du
    def computeCorrectionPatchDerTwo(self, isbdry, iv, iv2, patch, bndiff, u):
        if isbdry:
            coef = np.zeros(2, dtype=np.float64)
            jnd = np.zeros(2, dtype=int)
            jnd[0] = iv
            jnd[1] = iv2
            coef[0] += bndiff
            coef[1] -= bndiff
            return jnd, coef
        coef = np.zeros(4, dtype=np.float64)
        jnd = np.zeros(4, dtype=int)
        ii = patch[0][2]
        i0 = patch[1][ii][0]
        i1 = patch[1][ii][1]
        a0 = patch[2][ii][0]
        a1 = patch[2][ii][1]
        gradu = u[iv2] - u[iv]
        gradu2 = a0 * (u[i0] - u[iv]) + a1 * (u[i1] - u[iv])
        dx = bndiff*self.dnldx(gradu, gradu2)
        dy = bndiff*self.dnldy(gradu, gradu2)
        jnd[0] = iv
        jnd[1] = iv2
        jnd[2] = i0
        jnd[3] = i1
        coef[0] = dx + dy*(a0+a1)
        coef[1] = -dx
        coef[2] = -dy*a0
        coef[3] = -dy*a1
        return jnd, coef
    def matrixPatchesDiffusionTwo(self, u=None):
        nnodes =  self.mesh.nnodes
        nnpatches = self.nnpatches
        nplocal = 8
        index = np.zeros(nplocal * nnpatches, dtype=int)
        jndex = np.zeros(nplocal * nnpatches, dtype=int)
        A = np.zeros(nplocal * nnpatches, dtype=np.float64)
        count = 0
        for iv in range( nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            for ii, patch in self.mesh.patchinfo[iv].items():
                bndiff = self.sc[iv]*self.patchdiff[iv][ii]
                iv2 = patch[0][0]
                jnd, coef = self.computeCorrectionPatchDerTwo(boundary, iv, iv2, patch, bndiff, u)
                n = jnd.shape[0]
                n2 = n*2
                index[count:count+n] = iv
                index[count+n:count+n2] = iv2
                jndex[count:count+n] = jnd[:]
                jndex[count+n:count+n2] = jnd[:]
                A[count:count+n] += coef[:]
                A[count+n:count+n2] -= coef[:]
                count += n2
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def matrixPatchesDiffusionTwoScockCapturing(self, u):
        nnodes =  self.mesh.nnodes
        nmemory = nnodes*(self.npatchmax)**2*8
        index = np.zeros(nmemory, dtype=int)
        jndex = np.zeros(nmemory, dtype=int)
        A = np.zeros(nmemory, dtype=np.float64)
        count = 0
        mat = np.zeros(4, dtype=np.float64)
        js = np.zeros(4, dtype=int)
        for iv in range( nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            if boundary:
                continue
            g, g2 = self.computePatchGradients(iv, u)
            sc = self.phi(g, g2)
            dx, dy = self.dphi(g, g2)
            inds, phis = self.computeCorrectionPatchTwo(boundary, iv, sc, u)
            n = inds.shape[0]
            for ii in range(n):
                phi = phis[ii]
                iv2 = inds[ii]
                # du[iv] -= phi
                # du[iv2] += phi
                for jj, patch in self.mesh.patchinfo[iv].items():
                    jv2 = patch[0][0]
                    iop = patch[0][2]
                    j0 = patch[1][iop][0]
                    j1 = patch[1][iop][1]
                    a0 = patch[2][iop][0]
                    a1 = patch[2][iop][1]
                    js[0] = iv
                    js[1] = jv2
                    js[2] = j0
                    js[3] = j1
                    mat[0] = -dx[jj]-dy[ii]*(a0+a1)
                    mat[1] = dx[jj]
                    mat[2] = dy[jj]*a0
                    mat[3] = dy[jj]*a1
                    # mat[0] = dx[jj]
                    # mat[1] = -dx[jj]
                    # mat[2] = dy[jj]*a0
                    # mat[3] = dy[jj]*a1
                    mat *= phi
                    index[count:count+4] = iv
                    index[count+4:count+8] = iv2
                    jndex[count:count+4] = js[:]
                    # print 'js', js
                    # print 'jndex[count+4:count+8]', jndex[count+4:count+8]
                    jndex[count+4:count+8] = js
                    A[count:count+4] = mat[:]
                    A[count+4:count+8] = -mat[:]
                    count = count+8
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def computeCorrectionPatchTwoSym(self, isbdry, iv, iv2, patch, bndiff, du, u):
        gradu = u[iv2] - u[iv]
        gradcor = 0.0
        if isbdry:
            if self.bdrycor:
                xiv, yiv = self.mesh.x[iv], self.mesh.y[iv]
                xiv2, yiv2 = self.mesh.x[iv2], self.mesh.y[iv2]
                gradcor = self.dirichlet(xiv2, yiv2) - self.dirichlet(xiv, yiv)
        else:
            ii = patch[0][2]
            i0 = patch[1][ii][0]
            i1 = patch[1][ii][1]
            a0 = patch[2][ii][0]
            a1 = patch[2][ii][1]
            gradu2 = a0 * (u[i0] - u[iv]) +a1 * (u[i1] - u[iv])
            gradcor =  self.nl(gradu, gradu2)
            du[iv] -= (a0+a1)*bndiff * (gradu2 - gradcor)
            du[i0] += a0*bndiff * (gradu2 - gradcor)
            du[i1] += a1*bndiff * (gradu2 - gradcor)
        du[iv] -= bndiff * (gradu - gradcor)
        du[iv2] += bndiff * (gradu - gradcor)
    def formPatchesDiffusionTwoSym(self, du, u):
        for iv in range( self.mesh.nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            for ii,patch in self.mesh.patchinfo[iv].items():
                bndiff = self.patchdiff[iv][ii]
                iv2 = patch[0][0]
                self.computeCorrectionPatchTwoSym(boundary, iv, iv2, patch, bndiff, du, u)
        return du
    def computeCorrectionPatchDerTwoSym(self, isbdry, iv, iv2, patch, bndiff, u):
        coef = np.zeros(16, dtype=np.float64)
        ind = np.zeros(16, dtype=int)
        jnd = np.zeros(16, dtype=int)
        ind[0] = iv
        ind[1] = iv
        ind[2] = iv2
        ind[3] = iv2
        jnd[0] = iv
        jnd[1] = iv2
        jnd[2] = iv
        jnd[3] = iv2
        coef[0] += bndiff
        coef[1] -= bndiff
        coef[2] -= bndiff
        coef[3] += bndiff
        if isbdry:
            return ind, jnd, coef
        gradu = u[iv2] - u[iv]
        ii = patch[0][2]
        i0 = patch[1][ii][0]
        i1 = patch[1][ii][1]
        a0 = patch[2][ii][0]
        a1 = patch[2][ii][1]
        ind[4] = iv
        ind[5] = iv
        ind[6] = iv2
        ind[7] = iv2
        ind[8] = i0
        ind[9] = i0
        ind[10] = i0
        ind[11] = i0
        ind[12] = i1
        ind[13] = i1
        ind[14] = i1
        ind[15] = i1
        jnd[4] = i0
        jnd[5] = i1
        jnd[6] = i0
        jnd[7] = i1
        jnd[8] = iv
        jnd[9] = iv2
        jnd[10] = i0
        jnd[11] = i1
        jnd[12] = iv
        jnd[13] = iv2
        jnd[14] = i0
        jnd[15] = i1
        grad2 = a0 * (u[i0] - u[iv]) +a1 * (u[i1] - u[iv])
        dx = bndiff*self.dnldx(gradu, grad2)
        dy = bndiff*self.dnldy(gradu, grad2)
        dxdy01 = dx + dy * (a0 + a1)
        coef[4] += dy * a0
        coef[5] += dy * a1
        coef[0] -= dxdy01
        coef[6] -= dy * a0
        coef[7] -= dy * a1
        coef[2] += dxdy01
        coef[1] += dx
        coef[3] -= dx
        # sym
        coef[ 8] = a0 *  (  -bndiff *(a0+a1) + dxdy01    )
        coef[ 9] = a0 * (  -dx    )
        coef[10] = a0 *  (  bndiff *a0 -dy*a0   )
        coef[11] = a0 *  (  bndiff *a1 -dy*a1   )
        coef[12] = a1 * (  - bndiff *(a0+a1) + dxdy01   )
        coef[13] = a1 *  (  -dx    )
        coef[14] = a1 *  (  bndiff *a0 -dy*a0   )
        coef[15] = a1 *  (  bndiff *a1 -dy*a1   )

        coef[0] -= (a0+a1) * (  - bndiff *(a0+a1) +dxdy01    )
        coef[1] -= (a0+a1) *  (  -dx    )
        coef[4] -= (a0+a1) *  (  bndiff *a0 -dy*a0   )
        coef[5] -= (a0+a1) *  (  bndiff *a1 -dy*a1   )
        return ind, jnd, coef
    def matrixPatchesDiffusionTwoSym(self, u=None):
        nnodes =  self.mesh.nnodes
        nnpatches = self.nnpatches
        nplocal = 16
        index = np.zeros(nplocal * nnpatches, dtype=int)
        jndex = np.zeros(nplocal * nnpatches, dtype=int)
        A = np.zeros(nplocal * nnpatches, dtype=np.float64)
        count = 0
        for iv in range( nnodes):
            boundary = -1 in  self.mesh.patches_bnodes[iv][:, 5]
            for ii, patch in self.mesh.patchinfo[iv].items():
                bndiff = self.patchdiff[iv][ii]
                iv2 = patch[0][0]
                ind, jnd, coef = self.computeCorrectionPatchDerTwoSym(boundary, iv, iv2, patch, bndiff, u)
                index[count:count+nplocal] = ind[:]
                jndex[count:count+nplocal] = jnd[:]
                A[count:count+nplocal] += coef[:]
                count += nplocal
            # assert countbdry==self.mesh.nbdryvert
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def postProcess(self, u, timer, nit=True):
        point_data, cell_data, info = Transport.postProcess(self, u, timer, nit)
        point_data['sc'] = self.sc
        return point_data, cell_data, info



# ------------------------------------- #

if __name__ == '__main__':
    print('pas de main disponible')