# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from transportcg1 import TransportCg1
from tools.comparerrors import CompareErrors
from mesh.trimeshwithnodepatches import TriMeshWithNodePatches

class TransportCg1Edges(TransportCg1):
    """
    """
    def __init__(self, **kwargs):
        self.upwind = kwargs.pop('upwind')
        if kwargs.has_key('diff'):
            self.diff = kwargs.pop('diff')
        else:
            self.diff = 'min'
        TransportCg1.__init__(self, **kwargs)
        self.bdrycor =  kwargs['problem'].find('Analytic') >= 0

    def setMesh(self, mesh):
        TransportCg1.setMesh(self, mesh)
        self.mesh = TriMeshWithNodePatches(mesh)
        self.computeEdgeDiffusion()
        if self.upwind == 'nonlinear':
            self.computeEdgeInfo()
        if self.upwind == "nledgeup":
            self.computeEdgeUpwind()


    def computeEdgeInfo(self):
        nedges = self.mesh.nedges
        self.edgeinfo = (-np.ones((2,nedges,2), dtype=int), -np.ones((2,nedges,2), dtype=np.float64))
        for ie in range(nedges):
            il = self.mesh.edges[ie, 0]
            ir = self.mesh.edges[ie, 1]
            for ii in range(2):
                iv = self.mesh.edges[ie, ii]
                iinp = self.mesh.patch_indexofedge[ie, ii]
                if iinp >= 0:
                    i0 = self.mesh.patches_opind[iv][iinp, 0]
                    i1 = self.mesh.patches_opind[iv][iinp, 1]
                    a0 = self.mesh.patches_opcoeff[iv][iinp, 0]
                    a1 = self.mesh.patches_opcoeff[iv][iinp, 1]
                    if a0 > 0.0 or a1 > 0.0:
                        print ('positive coeff ', a0, a1, ' in ie', ie, ' iv=', iv, ' i0, i1', i0, i1)
                    self.edgeinfo[0][ii, ie, 0] = i0
                    self.edgeinfo[0][ii, ie, 1] = i1
                    self.edgeinfo[1][ii, ie, 0] = a0
                    self.edgeinfo[1][ii, ie, 1] = a1
                    if i0 < 0:
                        # bizarre, mais (par hazard ?) c'est vrai
                        assert ii==1
                        assert iv in self.mesh.bdryvert
                        # print 'no correction found for ie ', ie, '(ii=', ii, ') iv=', iv, ' i0, i1', i0, i1
                else:
                    print ('no correction found for ie', ie, ' and iv=', iv)
            # print 'ie', ie
            # print '\t il', self.mesh.edges[ie, 0], 'il0, il1', self.edgeinfo[0][0, ie]
            # print '\t ir', self.mesh.edges[ie, 1], 'ir0, ir1', self.edgeinfo[0][1, ie]
        # import matplotlib.pyplot as plt
        # self.mesh.plotPatches(plt)
    def computeEdgeUiipwind(self):
        nedges = self.mesh.nedges
        self.edgeupcoef = -np.ones((nedges, 3))
        self.edgeupind = -np.ones((nedges, 3), dtype=int)
        for ie in range(nedges):
            il = self.mesh.edges[ie, 0]
            ir = self.mesh.edges[ie, 1]
            xe = 0.5 * (self.mesh.x[self.mesh.edges[ie, 0]] + self.mesh.x[self.mesh.edges[ie, 1]])
            ye = 0.5 * (self.mesh.y[self.mesh.edges[ie, 0]] + self.mesh.y[self.mesh.edges[ie, 1]])
            beta = np.stack(self.beta(xe, ye), axis=-1)
            iis = -1
            onlyone = False
            for ii in range(2):
                iinp = self.mesh.patch_indexofedge[ie, ii]
                if iinp == -1:
                    onlyone = True
                    iis = 1 - ii
                    break
                iv = self.mesh.edges[ie, ii]
                i0 = self.mesh.patches_opind[iv][iinp, 0]
                i1 = self.mesh.patches_opind[iv][iinp, 1]
                if i0 == -1:
                    assert i1 == -1
                    onlyone = True
                    iis = 1 - ii
                    break
            if not onlyone:
                sp = np.zeros(2)
                for ii in range(2):
                    iinp = self.mesh.patch_indexofedge[ie, ii]
                    iv = self.mesh.edges[ie, ii]
                    i0 = self.mesh.patches_opind[iv][iinp, 0]
                    i1 = self.mesh.patches_opind[iv][iinp, 1]
                    a0 = self.mesh.patches_opcoeff[iv][iinp, 0]
                    a1 = self.mesh.patches_opcoeff[iv][iinp, 1]
                    x = -a0 * self.mesh.x[i0] - a1 * self.mesh.x[i1]
                    y = -a0 * self.mesh.y[i0] - a1 * self.mesh.y[i1]
                    dir = np.array((x - xe, y - ye))
                    sp[ii] = np.dot(dir, beta)
                iis = np.argmin(sp)
                if sp[iis] >= 0.0:
                    raise ValueError("not upwind %s" % str(sp))
            else:
                print("only one ! ie=%d" % ie)
                print ("patch_indexofedge", self.mesh.patch_indexofedge[ie])
                for ii in range(2):
                    iinp = self.mesh.patch_indexofedge[ie, ii]
                    if iinp == -1:
                        stop
                    iv = self.mesh.edges[ie, ii]
                    print ('iv', iv, 'patches_opind', self.mesh.patches_opind[iv][iinp])
                self.mesh.plotPatches(plt)
                raise ValueError("only one ! ie=%d" % ie)
                # print 'ie', ie, 'il', il, 'ir', ir, 'sp', sp,  self.mesh.edges[ie,ii]
            assert iis != -1
            iinp = self.mesh.patch_indexofedge[ie, iis]
            assert iinp != -1
            iv = self.mesh.edges[ie, iis]
            self.edgeupind[ie, 0] = self.mesh.patches_opind[iv][iinp, 0]
            self.edgeupind[ie, 1] = self.mesh.patches_opind[iv][iinp, 1]
            self.edgeupind[ie, 2] = iv
            self.edgeupcoef[ie, 0] = self.mesh.patches_opcoeff[iv][iinp, 0]
            self.edgeupcoef[ie, 1] = self.mesh.patches_opcoeff[iv][iinp, 1]
            self.edgeupcoef[ie, 2] = 2 * (iis == 0) - 1.0
    def form(self, du, u):
        self.formReaction(du, u)
        self.formEdges(du, u)
        self.formBoundary(du, u)
        return du
    def matrix(self, u=None):
        AR = self.matrixReaction()
        AI = self.matrixEdges(u)
        AB = self.matrixBoundary()
        A = (AI + AB + AR).tocsr()
        # tools.sparse.checkMmatrix(A)
        return A
    def formEdges(self, du, u):
        for ie in range( self.mesh.nedges):
            diffl = self.edgebeta[ie, 0]
            diffr = self.edgebeta[ie, 1]
            il =  self.mesh.edges[ie,0]
            ir =  self.mesh.edges[ie,1]
            udiff = u[ir]-u[il]
            du[il] += diffl*udiff
            du[ir] -= diffr*udiff
            if self.upwind=="centered": continue
            diff = self.edgediff[ie]
            du[il] -= diff * udiff
            du[ir] += diff * udiff
            if self.upwind == "linear": continue
            if ie in self.mesh.bdryedges:
                if self.bdrycor:
                    xil, yil = self.mesh.x[il], self.mesh.y[il]
                    xir, yir = self.mesh.x[ir], self.mesh.y[ir]
                    uDdiff = self.dirichlet(xir, yir) - self.dirichlet(xil, yil)
                    du[il] += diff * uDdiff
                    du[ir] -= diff * uDdiff
                continue
            il0 = self.edgeinfo[0][0, ie, 0]
            il1 = self.edgeinfo[0][0, ie, 1]
            al0 = self.edgeinfo[1][0, ie, 0]
            al1 = self.edgeinfo[1][0, ie, 1]
            ir0 = self.edgeinfo[0][1, ie, 0]
            ir1 = self.edgeinfo[0][1, ie, 1]
            ar0 = self.edgeinfo[1][1, ie, 0]
            ar1 = self.edgeinfo[1][1, ie, 1]
            if ir0 >= 0:
                diff *= 0.5
                udiffr = ar0*(u[ir]-u[ir0]) + ar1*(u[ir]-u[ir1])
                xir = -self.nl(-udiff, -udiffr)
                du[il] += diff * xir
                du[ir] -= diff * xir
            udiffl = al0*(u[il0]-u[il]) + al1*(u[il1]-u[il])
            xil = self.nl(udiff, udiffl)
            du[il] += diff * xil
            du[ir] -= diff * xil
        return du
    def matrixEdges(self, u=None):
        nnodes =  self.mesh.nnodes
        nedges =  self.mesh.nedges
        nlocal = 4
        if self.upwind == 'nonlinear':
            nlocal += 8
        index = np.zeros(nlocal*nedges, dtype=int)
        jndex = np.zeros(nlocal*nedges, dtype=int)
        A = np.zeros(nlocal*nedges, dtype=np.float64)
        count = 0
        for ie in range( self.mesh.nedges):
            diff0 = self.edgebeta[ie, 0]
            diff1 = self.edgebeta[ie, 1]
            il =  self.mesh.edges[ie,0]
            ir =  self.mesh.edges[ie,1]
            udiff = u[ir]-u[il]
            index[count+0] = il
            index[count+1] = il
            index[count+2] = ir
            index[count+3] = ir
            jndex[count+0] = il
            jndex[count+1] = ir
            jndex[count+2] = il
            jndex[count+3] = ir
            A[count + 0] -= diff0
            A[count + 1] += diff0
            A[count + 2] += diff1
            A[count + 3] -= diff1
            if self.upwind=="centered":
                count += nlocal
                continue
            diff = self.edgediff[ie]
            A[count + 0] += diff
            A[count + 1] -= diff
            A[count + 2] -= diff
            A[count + 3] += diff
            if self.upwind == "linear":
                count += nlocal
                continue
            if ie in self.mesh.bdryedges:
                count += nlocal
                continue
            il0 = self.edgeinfo[0][0, ie, 0]
            il1 = self.edgeinfo[0][0, ie, 1]
            al0 = self.edgeinfo[1][0, ie, 0]
            al1 = self.edgeinfo[1][0, ie, 1]
            ir0 = self.edgeinfo[0][1, ie, 0]
            ir1 = self.edgeinfo[0][1, ie, 1]
            ar0 = self.edgeinfo[1][1, ie, 0]
            ar1 = self.edgeinfo[1][1, ie, 1]
            index[count + 4] = il
            index[count + 5] = il
            index[count + 6] = ir
            index[count + 7] = ir
            jndex[count + 4] = il0
            jndex[count + 5] = il1
            jndex[count + 6] = il0
            jndex[count + 7] = il1
            if ir0 >= 0:
                diff *= 0.5
                index[count + 8] = il
                index[count + 9] = il
                index[count +10] = ir
                index[count +11] = ir
                jndex[count + 8] = ir0
                jndex[count + 9] = ir1
                jndex[count +10] = ir0
                jndex[count +11] = ir1
                udiffr = ar0 * (u[ir] - u[ir0]) + ar1 * (u[ir] - u[ir1])
                dxirdx = self.dnldx(-udiff, -udiffr)
                dxirdy = self.dnldy(-udiff, -udiffr)
                # xir = -self.nl(-udiff, -udiffr)
                # du[il] += diff * udiffr
                # du[ir] -= diff * udiffr
                A[count + 1] += (ar0+ar1)*diff*dxirdy
                A[count + 8] -= ar0* diff*dxirdy
                A[count + 9] -= ar1 * diff*dxirdy
                A[count + 3] -= (ar0 + ar1) * diff*dxirdy
                A[count +10] += ar0 * diff*dxirdy
                A[count +11] += ar1 * diff*dxirdy
                A[count + 0] -= diff*dxirdx
                A[count + 1] += diff*dxirdx
                A[count + 2] += diff*dxirdx
                A[count + 3] -= diff*dxirdx
            udiffl = al0 * (u[il0] - u[il]) + al1 * (u[il1] - u[il])
            dxildx = self.dnldx(udiff, udiffl)
            dxildy = self.dnldy(udiff, udiffl)
            # du[il] += diff * udiff2
            # du[ir] -= diff * udiff2
            A[count + 0] -= (al0+al1)*diff*dxildy
            A[count + 4] += al0*diff*dxildy
            A[count + 5] += al1*diff*dxildy
            A[count + 2] += (al0+al1)*diff*dxildy
            A[count + 6] -= al0*diff*dxildy
            A[count + 7] -= al1*diff*dxildy
            A[count + 0] -= diff*dxildx
            A[count + 1] += diff*dxildx
            A[count + 2] += diff*dxildx
            A[count + 3] -= diff*dxildx

            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))

    def formEdgeDiffusionUp(self, du, u):
        for ie in range( self.mesh.nedges):
            diff = self.edgediff[ie]
            il =  self.mesh.edges[ie,0]
            ir =  self.mesh.edges[ie,1]
            iv2 = self.edgeupind[ie, 0]
            iv3 = self.edgeupind[ie, 1]
            iv = self.edgeupind[ie, 2]
            a2 = self.edgeupcoef[ie, 0]
            a3 = self.edgeupcoef[ie, 1]
            sign = self.edgeupcoef[ie, 2]

            udiff = u[ir]-u[il]
            udiff2 =   a2*(u[iv2]-u[iv]) + a3*(u[iv3] -u[iv])
            xi = self.nl(sign*udiff, udiff2)
            dxidx = self.dnldx(sign*udiff, udiff2)
            dxidy = sign*self.dnldy(sign*udiff, udiff2)

            if self.sym == False:
                udiffcorr = udiff - sign*xi
                # udiffcorr = (1-dxidx)*udiff - dxidy*udiff2
                du[il] -= diff * udiffcorr
                du[ir] += diff * udiffcorr
            else:
                udiffcorr = udiff - sign*xi
                du[il] -= diff * (1-dxidx)*udiffcorr
                du[ir] += diff * (1-dxidx)*udiffcorr
                du[iv2] -= diff * dxidy * a2 *udiffcorr
                du[iv3] -= diff * dxidy * a3 *udiffcorr
                du[iv]  += diff * dxidy * (a2+a3) *udiffcorr
        return du
    def formEdgeDiffusion(self, du, u, nl=False):
        for ie in range( self.mesh.nedges):
            diff = self.edgediff[ie]
            il =  self.mesh.edges[ie,0]
            ir =  self.mesh.edges[ie,1]

            udiff = u[ir]-u[il]
            du[il] -= diff * udiff
            du[ir] += diff * udiff
            if nl:
                if ie in self.mesh.bdryedges:
                    assert il in self.mesh.bdryvert and ir in self.mesh.bdryvert
                xil, yil = self.mesh.x[il], self.mesh.y[il]
                xir, yir = self.mesh.x[ir], self.mesh.y[ir]
                udiffcorr = self.solexact(xir, yir) - self.solexact(xil, yil)
                du[il] += diff * udiffcorr
                du[ir] -= diff * udiffcorr
                continue

                iinp0 =  self.mesh.patch_indexofedge[ie, 0]
                iinp1 =  self.mesh.patch_indexofedge[ie, 1]
                i00 =  self.mesh.patches_opind[il][iinp0, 0]
                i01 =  self.mesh.patches_opind[il][iinp0, 1]
                a00 =  self.mesh.patches_opcoeff[il][iinp0][0]
                a01 =  self.mesh.patches_opcoeff[il][iinp0][1]
                i10 =  self.mesh.patches_opind[ir][iinp1, 0]
                i11 =  self.mesh.patches_opind[ir][iinp1, 1]
                a10 =  self.mesh.patches_opcoeff[ir][iinp1][0]
                a11 =  self.mesh.patches_opcoeff[ir][iinp1][1]
                if i00 == -1 and i10 == -1:
                    print ('*** no correction il', il, 'ir', ir, 'ie', ie)
                    print ('a', a00, a01, a10, a11)
                    print ('i', i00, i01, i10, i11)
                    print ('iinp', iinp0, iinp1)
                    self.mesh.testPatches()
                    self.mesh.plotPatches(plt)
                    raise ValueError("something's wrong")
                if i00 > -1 and i10 > -1:
                    diff *= 0.5
                # else:
                #     print "edge", ie, "only one correction il -- i00 i01 ", il, i00, i01, " ir -- i10 i11", ir, i10, i11
                if i00 > -1:
                    if a00 > 0.0 or a01 > 0.0:
                        raise ValueError(" a00 %g a01 %g" %(a00, a01))
                    udiff0 =   a00*(u[i00]-u[il])+a01*(u[i01] -u[il])
                    xi = self.nl(udiff, udiff0)
                    du[il] += diff * xi
                    du[ir] -= diff * xi
                if i10 > -1:
                    if a10 > 0.0 or a11 > 0.0:
                        raise ValueError(" a00 %g a01 %g" % (a10, a11))
                    udiff1 =   a10*(u[i10]-u[ir])+a11*(u[i11] -u[ir])
                    xi = self.nl(-udiff, udiff1)
                    du[il] -= diff * xi
                    du[ir] += diff * xi
        # self.mesh.plotPatches(plt)
        return du
    def matrixEdgeDiffusionUp(self, u=None):
        nnodes =  self.mesh.nnodes
        nedges =  self.mesh.nedges
        if self.sym == False:
            nlocal = 10
        else:
            nlocal = 25
        index = np.zeros(nlocal*nedges, dtype=int)
        jndex = np.zeros(nlocal*nedges, dtype=int)
        A = np.zeros(nlocal*nedges, dtype=np.float64)

        if self.sym == False:
            count = 0
            for ie in range( self.mesh.nedges):
                diff = self.edgediff[ie]
                il =  self.mesh.edges[ie,0]
                ir =  self.mesh.edges[ie,1]
                iv2 = self.edgeupind[ie, 0]
                iv3 = self.edgeupind[ie, 1]
                iv = self.edgeupind[ie, 2]
                a2 = self.edgeupcoef[ie, 0]
                a3 = self.edgeupcoef[ie, 1]
                sign = self.edgeupcoef[ie, 2]
                index[count] = il
                index[count+1] = il
                index[count+2] = ir
                index[count+3] = ir
                jndex[count+3] = ir
                index[count+4] = il
                index[count+5] = il
                index[count+6] = il
                index[count+7] = ir
                index[count+8] = ir
                index[count+9] = ir
                jndex[count] = il
                jndex[count+1] = ir
                jndex[count+2] = il
                jndex[count+4] = iv2
                jndex[count+5] = iv3
                jndex[count+6] = iv
                jndex[count+7] = iv2
                jndex[count+8] = iv3
                jndex[count+9] = iv

                udiff = u[ir] - u[il]
                udiff2 = a2 * (u[iv2] - u[iv]) + a3 * (u[iv3] - u[iv])
                dxidx = self.dnldx(sign*udiff, udiff2)
                dxidy = sign*self.dnldy(sign*udiff, udiff2)

                A[count + 0] += diff*(1 - dxidx)
                A[count + 1] -= diff*(1 - dxidx)
                A[count + 2] -= diff*(1 - dxidx)
                A[count + 3] += diff*(1 - dxidx)

                A[count + 4] += dxidy * diff*a2
                A[count + 5] += dxidy * diff*a3
                A[count + 6] -= dxidy * diff*(a2+a3)
                A[count + 7] -= dxidy * diff*a2
                A[count + 8] -= dxidy * diff*a3
                A[count + 9] += dxidy * diff*(a2+a3)
                count += nlocal
        else:
            count = 0
            for ie in range( self.mesh.nedges):
                diff = self.edgediff[ie]
                il =  self.mesh.edges[ie, 0]
                ir =  self.mesh.edges[ie, 1]
                iv2 = self.edgeupind[ie, 0]
                iv3 = self.edgeupind[ie, 1]
                iv = self.edgeupind[ie, 2]
                a2 = self.edgeupcoef[ie, 0]
                a3 = self.edgeupcoef[ie, 1]
                sign = self.edgeupcoef[ie, 2]
                index[count] = il
                index[count + 1] = il
                index[count + 2] = ir
                index[count + 3] = ir
                jndex[count + 3] = ir
                index[count + 4] = il
                index[count + 5] = il
                index[count + 6] = il
                index[count + 7] = ir
                index[count + 8] = ir
                index[count + 9] = ir
                jndex[count] = il
                jndex[count + 1] = ir
                jndex[count + 2] = il
                jndex[count + 4] = iv2
                jndex[count + 5] = iv3
                jndex[count + 6] = iv
                jndex[count + 7] = iv2
                jndex[count + 8] = iv3
                jndex[count + 9] = iv

                index[count + 10] = iv2
                index[count + 11] = iv2
                index[count + 12] = iv2
                index[count + 13] = iv2
                index[count + 14] = iv2

                index[count + 15] = iv3
                index[count + 16] = iv3
                index[count + 17] = iv3
                index[count + 18] = iv3
                index[count + 19] = iv3

                index[count + 20] = iv
                index[count + 21] = iv
                index[count + 22] = iv
                index[count + 23] = iv
                index[count + 24] = iv

                jndex[count + 10] = il
                jndex[count + 11] = ir
                jndex[count + 12] = iv2
                jndex[count + 13] = iv3
                jndex[count + 14] = iv

                jndex[count + 15] = il
                jndex[count + 16] = ir
                jndex[count + 17] = iv2
                jndex[count + 18] = iv3
                jndex[count + 19] = iv

                jndex[count + 20] = il
                jndex[count + 21] = ir
                jndex[count + 22] = iv2
                jndex[count + 23] = iv3
                jndex[count + 24] = iv

                udiff = u[ir] - u[il]
                udiff2 = a2 * (u[iv2] - u[iv]) + a3 * (u[iv3] - u[iv])
                dxidx = self.dnldx(sign * udiff, udiff2)
                dxidy = sign * self.dnldy(sign * udiff, udiff2)

                A[count + 0] += diff * (1 - dxidx)**2
                A[count + 1] -= diff * (1 - dxidx)**2
                A[count + 2] -= diff * (1 - dxidx)**2
                A[count + 3] += diff * (1 - dxidx)**2
                A[count + 4] += (1 - dxidx)*dxidy * diff * a2
                A[count + 5] += (1 - dxidx)*dxidy * diff * a3
                A[count + 6] -= (1 - dxidx)*dxidy * diff * (a2 + a3)
                A[count + 7] -= (1 - dxidx)*dxidy * diff * a2
                A[count + 8] -= (1 - dxidx)*dxidy * diff * a3
                A[count + 9] += (1 - dxidx)*dxidy * diff * (a2 + a3)

                A[count + 10] += diff*dxidy*a2 *(1-dxidx)
                A[count + 11] -= diff*dxidy*a2 *(1-dxidx)
                A[count + 12] += diff*dxidy*a2 *dxidy*a2
                A[count + 13] += diff*dxidy*a2 *dxidy*a3
                A[count + 14] -= diff*dxidy*a2 *dxidy*(a2+a3)

                A[count + 15] += diff*dxidy*a3 *(1-dxidx)
                A[count + 16] -= diff*dxidy*a3 *(1-dxidx)
                A[count + 17] += diff*dxidy*a3 *dxidy*a2
                A[count + 18] += diff*dxidy*a3 *dxidy*a3
                A[count + 19] -= diff*dxidy*a3 *dxidy*(a2+a3)

                A[count + 20] -= diff*dxidy*(a2+a3) *(1-dxidx)
                A[count + 21] += diff*dxidy*(a2+a3) *(1-dxidx)
                A[count + 22] -= diff*dxidy*(a2+a3) *dxidy*a2
                A[count + 23] -= diff*dxidy*(a2+a3) *dxidy*a3
                A[count + 24] += diff*dxidy*(a2+a3) *dxidy*(a2+a3)

                # du[iv2] -= diff * dxidy * a2 *( (1-dxidx)*udiff - dxidy*udiff2 )
                # du[iv3] -= diff * dxidy * a3 *( (1-dxidx)*udiff - dxidy*udiff2 )
                # du[iv]  += diff * dxidy * (a2+a3) *( (1-dxidx)*udiff - dxidy*udiff2 )



                count += nlocal

        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))
    def matrixEdgeDiffusion(self, u=None, nl=False):
        nnodes =  self.mesh.nnodes
        nedges =  self.mesh.nedges
        nlocal = 4
        if nl:
            nlocal += 8
        index = np.zeros(nlocal*nedges, dtype=int)
        jndex = np.zeros(nlocal*nedges, dtype=int)
        A = np.zeros(nlocal*nedges, dtype=np.float64)
        count = 0
        for ie in range( self.mesh.nedges):
            diff = self.edgediff[ie]
            il =  self.mesh.edges[ie,0]
            ir =  self.mesh.edges[ie,1]
            index[count] = il
            index[count+1] = il
            index[count+2] = ir
            index[count+3] = ir
            jndex[count] = il
            jndex[count+1] = ir
            jndex[count+2] = il
            jndex[count+3] = ir
            A[count + 0] += diff
            A[count + 1] -= diff
            A[count + 2] -= diff
            A[count + 3] += diff
            if nl:
                iinp0 =  self.mesh.patch_indexofedge[ie, 0]
                iinp1 =  self.mesh.patch_indexofedge[ie, 1]
                i00 =  self.mesh.patches_opind[il][iinp0, 0]
                i01 =  self.mesh.patches_opind[il][iinp0, 1]
                i10 =  self.mesh.patches_opind[ir][iinp1, 0]
                i11 =  self.mesh.patches_opind[ir][iinp1, 1]
                a00 =  self.mesh.patches_opcoeff[il][iinp0][0]
                a01 =  self.mesh.patches_opcoeff[il][iinp0][1]
                a10 =  self.mesh.patches_opcoeff[ir][iinp1][0]
                a11 =  self.mesh.patches_opcoeff[ir][iinp1][1]
                index[count+4] = il
                index[count+5] = il
                index[count+6] = il
                index[count+7] = il
                index[count+8] = ir
                index[count+9] = ir
                index[count+10] = ir
                index[count+11] = ir
                if i00 > -1 and i10 > -1:
                    diff *= 0.5
                if i00 > -1:
                    jndex[count+4] = i00
                    jndex[count+5] = i01
                    jndex[count+8] = i00
                    jndex[count+9] = i01
                if i10 > -1:
                    jndex[count+6] = i10
                    jndex[count+7] = i11
                    jndex[count+10] = i10
                    jndex[count+11] = i11

                if i00 > -1:
                # if i00 > -1 and i10 > -1:
                    udiff = u[ir] - u[il]
                    udiff0 =   a00*(u[i00]-u[il])+a01*(u[i01] -u[il])
                    dxidx = self.dnldx(udiff, udiff0)
                    dxidy = self.dnldy(udiff, udiff0)
                    A[count + 0] -= dxidy * diff*(a00+a01)
                    A[count + 4] += dxidy * diff*a00
                    A[count + 5] += dxidy * diff*a01
                    A[count + 2] += dxidy * diff*(a00+a01)
                    A[count + 8] -= dxidy * diff*a00
                    A[count + 9] -= dxidy * diff*a01
                    A[count + 0] -= dxidx * diff
                    A[count + 1] += dxidx * diff
                    A[count + 2] += dxidx * diff
                    A[count + 3] -= dxidx * diff

                if i10 > -1:
                # if i00 > -1 and i10 > -1:
                    udiff = u[il]-u[ir]
                    udiff1 =   a10*(u[i10]-u[ir])+a11*(u[i11] -u[ir])
                    dxidx = self.dnldx(udiff, udiff1)
                    dxidy = self.dnldy(udiff, udiff1)
                    A[count + 1] += dxidy * diff*(a10+a11)
                    A[count + 6] -= dxidy * diff*a10
                    A[count + 7] -= dxidy * diff*a11
                    A[count + 3] -= dxidy * diff*(a10+a11)
                    A[count +10] += dxidy * diff*a10
                    A[count +11] += dxidy * diff*a11
                    A[count + 0] -= dxidx * diff
                    A[count + 1] += dxidx * diff
                    A[count + 2] += dxidx * diff
                    A[count + 3] -= dxidx * diff
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes))


# ------------------------------------- #

if __name__ == '__main__':
    problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    # problem = 'Analytic_Exponential'
    problem = 'Analytic_Linear'
    # problem = 'RotStat'
    # problem = 'Ramp'
    alpha = 1.0
    alpha = 0.0
    # beta = lambda x, y: (-y, x)
    beta = lambda x, y: (-np.cos(np.pi * (x + y)), np.cos(np.pi * (x + y)))
    beta = lambda x, y: (-np.sin(np.pi * x) * np.cos(np.pi * y), np.sin(np.pi * y) * np.cos(np.pi * x))
    beta = lambda x, y: (-np.cos(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * y) * np.sin(np.pi * x))
    beta = None

    methods = {}
    methods['edgesC'] = TransportCg1Edges(upwind='centered', diff='min', problem=problem, alpha=alpha, beta=beta)
    methods['edgesL'] = TransportCg1Edges(upwind='linear', diff='min', problem=problem, alpha=alpha, beta=beta)
    # methods['edgesNlin'] = TransportCg1Edges(upwind='nonlinear', xi='xilin', diff='min', problem=problem, alpha=alpha, beta=beta)
    # methods['edgesN'] = TransportCg1Edges(upwind='nonlinear', xi='xi', diff='min', problem=problem, alpha=alpha, beta=beta)

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    h = [1.0, 0.5, 0.25, 0.1]
    # h = [1.0, 0.5, 0.25, 0.125]
    compareerrors.compare( h=h)
