# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(fempypath)
from raviartthomas import RaviartThomas
import fempy.tools.analyticalsolution
import fempy.fems.femrt0

class Laplace(RaviartThomas):
    """
    Fct de base de RT0 = sigma * 0.5 * |S|/|K| (x-x_N)
    """
    def __init__(self, **kwargs):
        RaviartThomas.__init__(self, **kwargs)
        self.dirichlet = None
        self.rhs = None
        self.solexact = None
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        self.femv = fempy.fems.femrt0.FemRT0()

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(mesh)

    def defineProblem(self, problem):
        self.problemname = problem
        problemsplit = problem.split('_')
        if problemsplit[0] == 'Analytic':
            if problemsplit[1] == 'Constant':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('7')
            elif problemsplit[1] == 'Linear':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x+2*y')
            elif problemsplit[1] == 'Quadratic':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
            elif problemsplit[1] == 'Hubbel':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
            elif problemsplit[1] == 'Exponential':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
            elif problemsplit[1] == 'Sinus':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
            else:
                raise ValueError("unknown analytic solution: {}".format(problemsplit[1]))
            self.dirichlet = solexact
            self.solexact = solexact
        else:
            raise ValueError("unownd problem {}".format(problem))

    def solve(self, iter, dirname):
        return self.solveLinear()

    def postProcess(self, u):
        nfaces =  self.mesh.nfaces
        info = {}
        cell_data = {'p': u[nfaces:]}
        vc = self.computeVCells(u[:nfaces])
        cell_data['v0'] = vc[0]
        cell_data['v1'] = vc[1]
        point_data = {}
        if self.solexact:
            err, pe, vex, vey = self.computeError(self.solexact, u, vc)
            cell_data['perr'] =np.abs(pe - u[nfaces:])
            cell_data['verrx'] =np.abs(vex - vc[0])
            cell_data['verry'] =np.abs(vey - vc[1])
            info['error'] = err
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        return point_data, cell_data, info

    def computeError(self, solexact, u, vc):
        nfaces =  self.mesh.nfaces
        errors = {}
        p = u[nfaces:]
        xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
        pex = solexact(xc, yc, zc)
        vexx = solexact.x(xc, yc, zc)
        vexy = solexact.y(xc, yc, zc)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.dV))
        errv = np.sqrt(np.sum( (vexx-vc[0])**2* self.mesh.dV +  (vexy-vc[1])**2* self.mesh.dV ))
        errors['pcL2'] = errp
        errors['vcL2'] = errv
        return errors, pex, vexx, vexy

    def computeRhs(self):
        xf, yf, zf = self.pointsf[:, 0], self.pointsf[:, 1], self.pointsf[:, 2]
        xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
        dirichlet, rhs = self.dirichlet, self.rhs
        nfaces =  self.mesh.nfaces
        bcells = None
        solexact = self.solexact
        if solexact:
            assert rhs is None
            bcells = (solexact.xx(xc, yc, zc) + solexact.yy(xc, yc, zc))* self.mesh.dV
        elif rhs:
            assert solexact is None
            bcells = rhs(xc, yc, zc) *  self.mesh.dV
        bsides = np.zeros(nfaces)
        for color, faces in self.mesh.bdrylabels.items():
            for ie in faces:
                ud = dirichlet(xf[ie], yf[ie], zf[ie])
                bsides[ie] = linalg.norm( self.mesh.normals[ie])*ud
        return np.concatenate((bsides, bcells))

    def matrix(self):
        import time
        t0 = time.time()
        AI = self.matrixInterior()
        AI = AI.tocsr()

        t1 = time.time()
        s0 = time.time()
        A = self.matrixA()
        s1 = time.time()
        A2 = self.femv.constructMass()
        s2 = time.time()
        print("A = ", s1-s0, " A2 = ", s2-s1)
        # print("A.data = ", A.data)
        # print("A2.data = ", A2.data)
        assert np.allclose(A.data, A2.data)

        # B = self.matrixB()
        B = self.femv.constructDiv()
        # print("B.data.shape = ", B.data.shape)
        # print("B2.data.shape = ", B2.data.shape)
        # assert np.allclose(B.data, B2.data)
        # print("B = ", B)
        A1 = scipy.sparse.hstack([A,B.T])
        ncells = self.mesh.ncells
        help = np.zeros((ncells))
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncells, ncells))
        A2 = scipy.sparse.hstack([B, help])
        A = scipy.sparse.vstack([A1,A2])
        A = A.tocsr()
        t2 = time.time()
        print("all = ", t1-t0, " sep = ", t2-t1)
        return A

    def boundary(self, A, b, u):
        return A,b,u

    def matrixInterior(self):
        ncells =  self.mesh.ncells
        nfaces =  self.mesh.nfaces
        elem =  self.mesh.simplices
        xf, yf, zf = self.pointsf[:, 0], self.pointsf[:, 1], self.pointsf[:, 2]
        xn, yn, zn = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
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
            edges =  self.mesh.facesOfCells[ic]
            for ii in range(3):
                iei = edges[ii]
                sigma[ii] = self.mesh.sigma[ic,ii]
                scale[ii] = 0.5*linalg.norm( self.mesh.normals[iei])/ self.mesh.dV[ic]
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = edges[ii]
                    jndex[count+3*ii+jj] = edges[jj]
            for kk in range(3):
                iek =  self.mesh.facesOfCells[ic, kk]
                for ii in range(3):
                    rt[ii,0] = sigma[ii] * scale[ii] * (xf[iek] -  xn[elem[ic, ii]])
                    rt[ii,1] = sigma[ii] * scale[ii] * (yf[iek] -  yn[elem[ic, ii]])
                for ii in range(3):
                    for jj in range(3):
                        A[count+3*ii+jj] +=  self.mesh.dV[ic]* np.dot(rt[ii], rt[jj])/3.0
            # p-psi v-chi
            index[count + 9] = ic + nfaces
            index[count + 10] = ic + nfaces
            index[count + 11] = ic + nfaces
            jndex[count + 12] = ic + nfaces
            jndex[count + 13] = ic + nfaces
            jndex[count + 14] = ic + nfaces
            for ii in range(3):
                ie = edges[ii]
                jndex[count+9+ii] = ie
                index[count+12+ii] = ie
                adiv = linalg.norm( self.mesh.normals[ie])* sigma[ii]
                A[count+9+ii] = adiv
                A[count+12+ii] = adiv
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nfaces+ncells, nfaces+ncells))

    def matrixA(self):
        ncells =  self.mesh.ncells
        nfaces =  self.mesh.nfaces
        elem =  self.mesh.simplices
        xf, yf, zf = self.pointsf[:, 0], self.pointsf[:, 1], self.pointsf[:, 2]
        xn, yn, zn = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
        nlocal = 9
        index = np.zeros(nlocal*ncells, dtype=int)
        jndex = np.zeros(nlocal*ncells, dtype=int)
        A = np.zeros(nlocal*ncells, dtype=np.float64)
        sigma = np.zeros(3, dtype=np.float64)
        scale = np.zeros(3, dtype=np.float64)
        rt = np.zeros((3,2), dtype=np.float64)
        count = 0
        for ic in range(ncells):
            # v-psi
            edges =  self.mesh.facesOfCells[ic]
            for ii in range(3):
                iei = edges[ii]
                sigma[ii] = self.mesh.sigma[ic,ii]
                scale[ii] = 0.5*linalg.norm( self.mesh.normals[iei])/ self.mesh.dV[ic]
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = edges[ii]
                    jndex[count+3*ii+jj] = edges[jj]
            for kk in range(3):
                iek =  self.mesh.facesOfCells[ic, kk]
                for ii in range(3):
                    rt[ii,0] = sigma[ii] * scale[ii] * (xf[iek] -  xn[elem[ic, ii]])
                    rt[ii,1] = sigma[ii] * scale[ii] * (yf[iek] -  yn[elem[ic, ii]])
                for ii in range(3):
                    for jj in range(3):
                        A[count+3*ii+jj] +=  self.mesh.dV[ic]* np.dot(rt[ii], rt[jj])/3.0
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nfaces, nfaces))

    # def matrixB(self):
    #     ncells =  self.mesh.ncells
    #     nfaces =  self.mesh.nfaces
    #     nlocal = 3
    #     index = np.zeros(nlocal*ncells, dtype=int)
    #     jndex = np.zeros(nlocal*ncells, dtype=int)
    #     A = np.zeros(nlocal*ncells, dtype=np.float64)
    #     count = 0
    #     for ic in range(ncells):
    #         for ii in range(3):
    #             ie = self.mesh.facesOfCells[ic,ii]
    #             index[count+ii] = ic
    #             jndex[count+ii] = ie
    #             adiv = linalg.norm( self.mesh.normals[ie])* self.mesh.sigma[ic,ii]
    #             A[count+ii] = adiv
    #         count += nlocal
    #     print("*** A.shape", A.shape)
    #     A= scipy.sparse.coo_matrix((A, (index, jndex)), shape=(ncells, nfaces))
    #     # print("A",A.todense())
    #     return A

# ------------------------------------- #

if __name__ == '__main__':
    import fempy.tools.comparerrors
    import sys
    problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    # problem = 'Analytic_Linear'
    # problem = 'Analytic_Constant'

    methods = {}
    methods['poisson'] = Laplace(problem=problem)

    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [2, 1.0, 0.5, 0.25, 0.125, 0.062]
    comp.compare(h=h)
