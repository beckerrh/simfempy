# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse
import matplotlib.pyplot as plt

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from tools.analyticalsolution import AnalyticalSolution
from fem.femp12d import FemP12D
from mesh.trimesh import TriMesh
from tools.comparerrors import CompareErrors
from tools.solver import Solver
from  raviartthomas import RaviartThomas


class Euler(RaviartThomas):
    def __init__(self, problem=None, method="centered", alpha=100.0):
        RaviartThomas.__init__(self)
        self.problem = problem
        self.method = method
        self.alpha = alpha
        self.dirichlet = None
        self.rhs = None
        self.solexact = None
        self.defineProblem()

    def defineProblem(self):
        assert problem is not None
        problemsplit = self.problem.split('_')
        if problemsplit[0] == 'Analytic':
            if problemsplit[1] == 'Linear':
                solexactp = AnalyticalSolution('0.3 * x + 0.7 * y')
                solexactv1 = AnalyticalSolution('1.0 + 0.3 * x + 0.7 * y')
                solexactv2 = AnalyticalSolution('-1.0 + 0.7 * x - 0.3 * y')
                # solexactv1 = AnalyticalSolution('-0.3')
                # solexactv2 = AnalyticalSolution('-0.7')
            elif problemsplit[1] == 'Quadratic':
                solexactp = AnalyticalSolution('x*x+2*y*y')
                solexactv1 = AnalyticalSolution('1.0 + 0.3 * x + 0.7 * y')
                solexactv2 = AnalyticalSolution('-1.0 + 0.7 * x - 0.3 * y')
            else:
                raise ValueError("unknown analytic solution: '%s'" %(problemsplit[1]))
            self.solexact = (solexactp, solexactv1, solexactv2)
            self.dirichlet = self.solexact
        else:
            self.solexact = None
            if problem == 'Canal':
                dirichletp = AnalyticalSolution('0.0')
                # dirichletv1 = AnalyticalSolution('1-y**2')
                dirichletv1 = AnalyticalSolution('1.0')
                dirichletv2 = AnalyticalSolution('0.0')
                self.dirichlet = (dirichletp, dirichletv1, dirichletv2)
            else:
                raise ValueError("unownd problem %s" %problem)

    def solve(self):
        u = self.initialize()
        # return self.solveLinear()
        return self.solveNonlinear(u)

    def initialize(self):
        (solexactp, solexactv1, solexactv2) = self.dirichlet
        nedges =  self.mesh.nedges
        ncells =  self.mesh.ncells
        u = np.zeros(nedges+ncells)
        p = u[nedges:]
        for ic in range(ncells):
            p[ic] = solexactp( self.mesh.centersx[ic], self.mesh.centersy[ic] )
        normals =  self.mesh.normals
        xe = self.xedge
        ye = self.yedge
        for ie in range(nedges):
            h = linalg.norm(normals[ie])
            u[ie] = (solexactv1(xe[ie], ye[ie])*normals[ie,0] + solexactv2(xe[ie], ye[ie])*normals[ie,1])/h
        return u

    def postProcess(self, u, timer, nit=True, vtkbeta=True):
        nedges =  self.mesh.nedges
        info = {}
        cell_data = {'p': u[nedges:]}
        # v0,v1 = self.computeVCells(u[:nedges])
        v0,v1 = self.computeVEdges(u[:nedges])
        cell_data['v0'] = v0
        cell_data['v1'] = v1
        point_data = {}
        if self.solexact:
            err, pe, vex, vey = self.computeError(self.solexact, u, (v0, v1))
            cell_data['pex'] =  pe
            cell_data['perr'] =np.abs(pe - u[nedges:])
            cell_data['verrx'] =np.abs(vex - v0)
            cell_data['verry'] =np.abs(vey - v1)
            info['error'] = err
        # info['timer'] = self.timer
        if nit: info['nit'] = nit
        return point_data, cell_data, info

    def computeError(self, solexact, u, vcell):
        nedges =  self.mesh.nedges
        ncells =  self.mesh.ncells
        (solexactp, solexactv1, solexactv2) = solexact
        errors = {}
        p = u[nedges:]
        vx, vy = vcell
        pex = solexactp(self.mesh.centerx, self.mesh.centery)
        vexx = solexactv1(self.mesh.centerx, self.mesh.centery)
        vexy = solexactv2(self.mesh.centerx, self.mesh.centery)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.area))
        errv = np.sqrt(np.sum( (vexx-vx)**2* self.mesh.area +  (vexy-vy)**2* self.mesh.area ))
        errors['pL2'] = errp
        errors['vL2'] = errv
        return errors, pex, vexx, vexy

    def computeRhs(self):
        (dirichletp, dirichletv1, dirichletv2) = self.dirichlet
        nedges =  self.mesh.nedges
        ncells =  self.mesh.ncells
        bdryedges =  self.mesh.bdryedges
        nbdryedges = len(bdryedges)
        elem =  self.mesh.triangles
        bcells = None
        solexact = self.solexact
        xmid, ymid =  self.mesh.centersx,  self.mesh.centersy
        bsides = np.zeros(nedges)
        if solexact:
            assert self.rhs is None
            (solexactp, solexactv1, solexactv2) = solexact
            bcells = (solexactv1.x(xmid, ymid) + solexactv2.y(xmid, ymid))* self.mesh.area[:]
            bcells[:]=0
            for ic in range(ncells):
                edges =  self.mesh.edgesOfCell[ic]
                for kk in range(3):
                    iek =  self.mesh.edgesOfCell[ic, kk]
                    xk = self.xedge[iek]
                    yk = self.yedge[iek]
                    v1 = solexactv1(xk, yk)
                    v2 = solexactv2(xk, yk)
                    f1 = self.alpha*v1 + v1*solexactv1.x(xk, yk) + v2*solexactv1.y(xk, yk) + solexactp.x(xk,yk)
                    f2 = self.alpha*v2 + v1*solexactv2.x(xk, yk) + v2*solexactv2.y(xk, yk) + solexactp.y(xk,yk)
                    rt = self.rt(ic, xk, yk)
                    for ii in range(3):
                       bsides[edges[ii]] += ( self.mesh.area[ic]/3.0)*( f1*rt[ii,0] + f2*rt[ii,1])
        elif self.rhs:
            assert solexact is None
            (rhsp, rhsv1, rhsv2) = self.rhs
            bcells = rhsp(xmid, ymid) *  self.mesh.area[:]
            assert 0
        else:
            bcells = np.zeros(ncells)

        # for (count, ie) in enumerate(bdryedges):
        #     sigma = 1.0
        #     if  self.mesh.cellsOfEdge[ie, 0]==-1:
        #         sigma = -1.0
        #     pd = dirichletp(self.xedge[ie], self.yedge[ie])
        #     bsides[ie] -= linalg.norm( self.mesh.normals[ie])*sigma*pd
        return np.concatenate((bsides, bcells))

    def rtns(self, ic, x, y):
        (rt, rtgrad) = self.rtpgrad(ic, x, y)
        tform = np.zeros((3,3,3), dtype=np.float64)
        for i in range(3):
            for p in range(3):
                for q in range(3):
                    for l in range(2):
                        # tform[i,p,q] += 0.5*rt[p,l]*( rtgrad[q]*rt[i,l] -  rtgrad[i]*rt[q,l] )
                        tform[i, p, q] +=rt[p, l] * rtgrad[q] * rt[i, l]
        return tform

    def form(self, du, u):
        self.formCells(du, u)
        self.formInteriorSides(du, u)
        self.formBoundarySides(du, u)
        return du

    def formBoundarySides(self, du, u):
        (dirichletp, dirichletv1, dirichletv2) = self.dirichlet
        nedges =  self.mesh.nedges
        for ie in  self.mesh.bdryedges:
            vn = u[ie]
            ic =  self.mesh.cellsOfEdge[ie]
            h = linalg.norm( self.mesh.normals[ie])
            xe, ye = self.xedge[ie], self.yedge[ie]
            sigma = 1.0
            if ic[0]==-1:
                vn *= -1.0
                sigma = -1.0
            if vn >= 0.0:
                pd = dirichletp(xe, ye)
                du[ie] += h * sigma * pd
                continue

            ic = ic[np.where(ic!=-1)]
            du[ie] -= h * sigma * u[nedges + ic[0]]

            rt0 = self.rt(ic[0], xe, ye)
            ie0 =  self.mesh.edgesOfCell[ic[0], :]
            vx0 = np.dot( rt0[:, 0] , u[ie0] )
            vy0 = np.dot( rt0[:, 1] , u[ie0])
            vxD = dirichletv1(xe, ye)
            vyD = dirichletv2(xe, ye)
            for ii in range(3):
                du[ie0[ii]] -= h*vn * ((vx0-vxD)*rt0[ii,0] + (vy0-vyD)*rt0[ii,1] )
        return du

    def formInteriorSides(self, du, u):
        nedges =  self.mesh.nedges
        v = u[:nedges]
        dv = du[:nedges]
        nedges =  self.mesh.nedges
        for ie in range(nedges):
            ic =  self.mesh.cellsOfEdge[ie]
            ic = ic[np.where(ic!=-1)]
            if len(ic)==1:
                continue
            vn = v[ie]
            h = linalg.norm( self.mesh.normals[ie])
            xe, ye = self.xedge[ie], self.yedge[ie]
            rt0 = self.rt(ic[0], xe, ye)
            ie0 =  self.mesh.edgesOfCell[ic[0], :]
            vx0 = np.dot( rt0[:, 0] , u[ie0] )
            vy0 = np.dot( rt0[:, 1] , u[ie0])
            rt1 = self.rt(ic[1], xe, ye)
            ie1 =  self.mesh.edgesOfCell[ic[1], :]
            vx1 = np.dot(rt1[:, 0], u[ie1])
            vy1 = np.dot(rt1[:, 1], u[ie1])
            vjumpx = vx0 - vx1
            vjumpy = vy0 - vy1

            if self.method=="centered":
                for ii in range(3):
                    dv[ie0[ii]] -= 0.5*h*vn * (vjumpx * rt0[ii, 0] + vjumpy * rt0[ii, 1])
                    dv[ie1[ii]] -= 0.5*h*vn * (vjumpx * rt1[ii, 0] + vjumpy * rt1[ii, 1])
            else:
                if vn > 0.0:
                    for ii in range(3):
                        dv[ie1[ii]] -= h*vn*(vjumpx * rt1[ii, 0] + vjumpy * rt1[ii, 1])
                elif vn < 0.0:
                    for ii in range(3):
                        dv[ie0[ii]] -= h*vn*(vjumpx*rt0[ii,0] + vjumpy*rt0[ii,1])

        return du

    def formCells(self, du, u):
        nedges =  self.mesh.nedges
        v = u[:nedges]
        p = u[nedges:]
        dv = du[:nedges]
        dp = du[nedges:]

        ncells =  self.mesh.ncells
        sigma = np.zeros(3, dtype=np.float64)
        for ic in range(ncells):
            edges =  self.mesh.edgesOfCell[ic]
            vedges = v[edges]
            for kk in range(3):
                iek = edges[kk]
                xk = self.xedge[iek]
                yk = self.yedge[iek]
                rt = self.rt(ic, xk, yk)
                vk = np.inner(rt.T, vedges)
                for ii in range(3):
                    ie = edges[ii]
                    dv[ie] += self.alpha * ( self.mesh.area[ic]/3.0) * np.dot(vk, rt[ii] )
            for ii in range(3):
                ie = edges[ii]
                sigma = 2.0*( self.mesh.cellsOfEdge[ie,0]==ic)-1.0
                adiv = linalg.norm( self.mesh.normals[ie])* sigma
                dp[ic] += adiv* v[ie]
                dv[ie] -= adiv* p[ic]
        return du

    def matrix(self, u = None):
        A = self.matrixCells(u)
        if self.method == "centered":
            A += self.matrixInteriorSidesCentered(u)
        else:
            A += self.matrixInteriorSidesUpwind(u)
        A += self.matrixBoundarySides(u)
        return A.tocsr()

    def matrixInteriorSidesCentered(self, u):
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        nintedges =  self.mesh.nintedges
        v = u[:nedges]
        nlocal = 42
        index = np.zeros(nlocal*nintedges, dtype=int)
        jndex = np.zeros(nlocal*nintedges, dtype=int)
        A = np.zeros(nlocal*nintedges, dtype=np.float64)
        count = 0
        for ie in range(nedges):
            ic =  self.mesh.cellsOfEdge[ie]
            ic = ic[np.where(ic!=-1)]
            if len(ic)==1:
                continue
            vn = v[ie]
            h = linalg.norm( self.mesh.normals[ie])
            xe, ye = self.xedge[ie], self.yedge[ie]
            rt0 = self.rt(ic[0], xe, ye)
            ie0 =  self.mesh.edgesOfCell[ic[0], :]
            vx0 = np.dot( rt0[:, 0] , u[ie0] )
            vy0 = np.dot( rt0[:, 1] , u[ie0])
            rt1 = self.rt(ic[1], xe, ye)
            ie1 =  self.mesh.edgesOfCell[ic[1], :]
            vx1 = np.dot(rt1[:, 0], u[ie1])
            vy1 = np.dot(rt1[:, 1], u[ie1])
            vjumpx = vx0 - vx1
            vjumpy = vy0 - vy1
            sigma0 = 2.0 * ( self.mesh.cellsOfEdge[ie, 0] == ic[0]) - 1.0
            sigma1 = 2.0 * ( self.mesh.cellsOfEdge[ie, 0] == ic[1]) - 1.0
            for ii in range(3):
                for jj in range(3):
                    index[count + 3*ii + jj] = ie0[ii]
                    index[count + 9 + 3*ii + jj] = ie0[ii]
                    index[count + 18 + 3*ii + jj] = ie1[ii]
                    index[count + 27 + 3*ii + jj] = ie1[ii]
                    jndex[count + 3*ii + jj] = ie0[jj]
                    jndex[count + 9 + 3*ii + jj] = ie1[jj]
                    jndex[count + 18 + 3*ii + jj] = ie0[jj]
                    jndex[count + 27 + 3*ii + jj] = ie1[jj]
                    A[count + 3*ii + jj]      -= 0.5*h*vn*( rt0[jj, 0] * rt0[ii, 0] + rt0[jj, 1] * rt0[ii, 1] )
                    A[count + 9 + 3*ii + jj]  += 0.5*h*vn*( rt1[jj, 0] * rt0[ii, 0] + rt1[jj, 1] * rt0[ii, 1] )
                    A[count + 18 + 3*ii + jj] -= 0.5*h*vn*( rt0[jj, 0] * rt1[ii, 0] + rt0[jj, 1] * rt1[ii, 1] )
                    A[count + 27 + 3*ii + jj] += 0.5*h*vn*( rt1[jj, 0] * rt1[ii, 0] + rt1[jj, 1] * rt1[ii, 1] )
            for ii in range(3):
                index[count + 36 + ii] = ie0[ii]
                index[count + 39 + ii] = ie1[ii]
                jndex[count + 36 + ii] = ie
                jndex[count + 39 + ii] = ie
                A[count + 36 + ii] -= 0.5 * h *  (vjumpx * rt0[ii, 0] + vjumpy * rt0[ii, 1])
                A[count + 39 + ii] -= 0.5 * h *  (vjumpx * rt1[ii, 0] + vjumpy * rt1[ii, 1])
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges+ncells, nedges+ncells))

    def matrixInteriorSidesUpwind(self, u):
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        nintedges =  self.mesh.nintedges
        v = u[:nedges]
        nlocal = 21
        index = np.zeros(nlocal*nintedges, dtype=int)
        jndex = np.zeros(nlocal*nintedges, dtype=int)
        A = np.zeros(nlocal*nintedges, dtype=np.float64)
        count = 0
        for ie in range(nedges):
            ic =  self.mesh.cellsOfEdge[ie]
            ic = ic[np.where(ic!=-1)]
            if len(ic)==1:
                continue
            vn = v[ie]
            h = linalg.norm( self.mesh.normals[ie])
            xe, ye = self.xedge[ie], self.yedge[ie]
            rt0 = self.rt(ic[0], xe, ye)
            ie0 =  self.mesh.edgesOfCell[ic[0], :]
            vx0 = np.dot( rt0[:, 0] , u[ie0] )
            vy0 = np.dot( rt0[:, 1] , u[ie0])
            rt1 = self.rt(ic[1], xe, ye)
            ie1 =  self.mesh.edgesOfCell[ic[1], :]
            vx1 = np.dot(rt1[:, 0], u[ie1])
            vy1 = np.dot(rt1[:, 1], u[ie1])
            vjumpx = vx0 - vx1
            vjumpy = vy0 - vy1
            sigma0 = 2.0 * ( self.mesh.cellsOfEdge[ie, 0] == ic[0]) - 1.0
            sigma1 = 2.0 * ( self.mesh.cellsOfEdge[ie, 0] == ic[1]) - 1.0
            if vn > 0.0:
                for ii in range(3):
                    for jj in range(3):
                        index[count + 0 + 3*ii + jj] = ie1[ii]
                        index[count + 9 + 3*ii + jj] = ie1[ii]
                        jndex[count + 0 + 3*ii + jj] = ie0[jj]
                        jndex[count + 9 + 3*ii + jj] = ie1[jj]
                        A[count + 0 + 3 * ii + jj] -=  h * vn * (rt0[jj, 0] * rt1[ii, 0] + rt0[jj, 1] * rt1[ii, 1])
                        A[count + 9 + 3 * ii + jj] +=  h * vn * (rt1[jj, 0] * rt1[ii, 0] + rt1[jj, 1] * rt1[ii, 1])
                for ii in range(3):
                    index[count + 18 + ii] = ie1[ii]
                    jndex[count + 18 + ii] = ie
                    A[count + 18 + ii] -= h *  (vjumpx * rt1[ii, 0] + vjumpy * rt1[ii, 1])
            elif vn < 0.0:
                for ii in range(3):
                    for jj in range(3):
                        index[count + 0 + 3*ii + jj] = ie0[ii]
                        index[count + 9 + 3*ii + jj] = ie0[ii]
                        jndex[count + 0 + 3*ii + jj] = ie0[jj]
                        jndex[count + 9 + 3*ii + jj] = ie1[jj]
                        A[count + 3*ii + jj]      -= h*vn*( rt0[jj, 0] * rt0[ii, 0] + rt0[jj, 1] * rt0[ii, 1] )
                        A[count + 9 + 3*ii + jj]  += h*vn*( rt1[jj, 0] * rt0[ii, 0] + rt1[jj, 1] * rt0[ii, 1] )
                for ii in range(3):
                    index[count + 18 + ii] = ie0[ii]
                    jndex[count + 18 + ii] = ie
                    A[count + 18 + ii] -= h *  (vjumpx * rt0[ii, 0] + vjumpy * rt0[ii, 1])
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges+ncells, nedges+ncells))

    def matrixBoundarySides(self, u):
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        nbdryedges =  self.mesh.nbdryedges
        v = u[:nedges]
        nlocal = 13
        index = np.zeros(nlocal*nbdryedges, dtype=int)
        jndex = np.zeros(nlocal*nbdryedges, dtype=int)
        A = np.zeros(nlocal*nbdryedges, dtype=np.float64)
        count = 0
        (dirichletp, dirichletv1, dirichletv2) = self.dirichlet
        for ie in  self.mesh.bdryedges:
            vn = u[ie]
            ic =  self.mesh.cellsOfEdge[ie]
            sigma = 1.0
            if ic[0]==-1:
                vn *= -1.0
                sigma = -1.0
            if vn >= 0.0:
                count += nlocal
                continue
            ic = ic[np.where(ic!=-1)]
            h = linalg.norm( self.mesh.normals[ie])
            xe, ye = self.xedge[ie], self.yedge[ie]
            rt0 = self.rt(ic[0], xe, ye)
            ie0 =  self.mesh.edgesOfCell[ic[0], :]
            vx0 = np.dot( rt0[:, 0] , u[ie0] )
            vy0 = np.dot( rt0[:, 1] , u[ie0])
            vxD = dirichletv1(xe, ye)
            vyD = dirichletv2(xe, ye)

            for ii in range(3):
                for jj in range(3):
                    index[count + 3 * ii + jj] = ie0[ii]
                    jndex[count + 3 * ii + jj] = ie0[jj]
                    A[count + 3 * ii + jj] -= h * vn * (rt0[jj, 0] * rt0[ii, 0] + rt0[jj, 1] * rt0[ii, 1])
            for ii in range(3):
                index[count + 9 + ii] = ie0[ii]
                jndex[count + 9 + ii] = ie
                A[count + 9 + ii] -= sigma * h *  ((vx0-vxD)*rt0[ii,0] + (vy0-vyD)*rt0[ii,1] )
            index[count + 12] = ie
            jndex[count + 12] = nedges + ic[0]
            A[count + 12] -= h * sigma
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges+ncells, nedges+ncells))

    def matrixCells(self, u):
        """
        on suppose que  self.mesh.edgesOfCell[ic, kk] et oppose à elem[ic,kk] !!!
        """
        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        elem =  self.mesh.triangles

        nlocal = 15
        index = np.zeros(nlocal*ncells, dtype=int)
        jndex = np.zeros(nlocal*ncells, dtype=int)
        A = np.zeros(nlocal*ncells, dtype=np.float64)
        edges = np.zeros(3, dtype=int)
        sigma = np.zeros(3, dtype=np.float64)
        scale = np.zeros(3, dtype=np.float64)
        rt = np.zeros((3,2), dtype=np.float64)
        count = 0
        for ic in range(ncells):
            # v-psi
            edges =  self.mesh.edgesOfCell[ic]
            for ii in range(3):
                iei = edges[ii]
                sigma[ii] = 2.0*( self.mesh.cellsOfEdge[iei,0]==ic)-1.0
                scale[ii] = 0.5*linalg.norm( self.mesh.normals[iei])/ self.mesh.area[ic]
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = edges[ii]
                    jndex[count+3*ii+jj] = edges[jj]
            for kk in range(3):
                iek = edges[kk]
                xk = self.xedge[iek]
                yk = self.yedge[iek]
                rt = self.rt(ic, xk, yk)
                for ii in range(3):
                    for jj in range(3):
                        A[count+3*ii+jj] += self.alpha *  self.mesh.area[ic]* np.dot(rt[ii], rt[jj])/3.0
            # p-psi v-chi
            index[count + 9] = ic + nedges
            index[count + 10] = ic + nedges
            index[count + 11] = ic + nedges
            jndex[count + 12] = ic + nedges
            jndex[count + 13] = ic + nedges
            jndex[count + 14] = ic + nedges
            for ii in range(3):
                ie = edges[ii]
                jndex[count+9+ii] = ie
                index[count+12+ii] = ie
                adiv = linalg.norm( self.mesh.normals[ie])* sigma[ii]
                A[count+9+ii] = adiv
                A[count+12+ii] = -adiv
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges+ncells, nedges+ncells))

# ------------------------------------- #

if __name__ == '__main__':
    problem = 'Analytic_Linear'
    problem = 'Analytic_Quadratic'
    # problem = 'Canal'

    alpha = 0.0
    methods = {}
    # methods['eulerce'] = Euler(problem=problem, alpha=alpha)
    methods['eulerup'] = Euler(problem=problem, method="upwind", alpha=alpha)

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    h = [1.0, 0.5, 0.25, 0.125, 0.06]
    # h = [1.0, 0.5, 0.25]
    compareerrors.compare(h=h)
