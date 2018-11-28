# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.integrate

from mesh.trimesh import TriMesh
from tools.analyticalsolution import AnalyticalSolution
from tools.solver import Solver
import time
from tools.newton import newton
import scipy.sparse.linalg as splinalg


class Transport(Solver):
    def __init__(self, **kwargs):
        Solver.__init__(self)
        if 'finaltime' in kwargs:
            self.finaltime = kwargs.pop('finaltime')
            self.nvisusteps = 100
        else:
            self.finaltime=None
        if 'dontcomputeBcells' in kwargs:
            self.dontcomputeBcells = kwargs.pop('dontcomputeBcells')
        else:
            self.dontcomputeBcells = False
        if 'errorformula' in kwargs:
            self.errorformula = kwargs.pop('errorformula')
        else:
            self.errorformula = "node"
        if 'mesh' in kwargs:
            self.mesh = kwargs.pop('mesh')
        else:
            self.mesh = None
        if 'beta' in kwargs:
            self.beta = kwargs.pop('beta')
        else:
            self.beta = None
        if 'alpha' in kwargs:
            self.alpha = kwargs.pop('alpha')
        else:
            self.alpha = 0.0
        if 'dirichlet' in kwargs:
            self.dirichlet = kwargs.pop('dirichlet')
        else:
            self.dirichlet = None
        if 'rhs' in kwargs:
            self.rhs = kwargs.pop('rhs')
        else:
            self.rhs = None
        if 'solexact' in kwargs:
            self.solexact = kwargs.pop('solexact')
        else:
            self.solexact = None
        if 'initialcondition' in kwargs:
            self.initialcondition = kwargs.pop('initialcondition')
        else:
            self.initialcondition = None
        if 'problem' in kwargs:
            self.problem = kwargs.pop('problem')
        else:
            self.problem = "none"

        self.defineProblem()
        self.betan = None
        self.vtkbeta = True
        if self.mesh:
            self.setMesh(self.mesh)
        self.timer = {'rhs':0.0, 'matrix':0.0, 'solve':0.0}
    def setMesh(self, mesh):
        self.mesh = mesh
        self.betan = np.zeros( self.mesh.nedges)
        def betafct(p, x0, x1, y0, y1, normal):
            x = p * x0 + (1.0 - p) * x1
            y = p * y0 + (1.0 - p) * y1
            beta = np.stack(self.beta(x,y))
            return np.dot(normal, beta)
        for ie in range( self.mesh.nedges):
            i0 = self.mesh.edges[ie, 0]
            i1 = self.mesh.edges[ie, 1]
            x0, x1 = self.mesh.x[i0], self.mesh.x[i1]
            y0, y1 = self.mesh.y[i0], self.mesh.y[i1]
            self.betan[ie] = scipy.integrate.quadrature(betafct, 0, 1, args=(x0, x1, y0, y1,self.mesh.normals[ie]), tol=1.e-10, rtol=1.e-10)[0]
        for it in range( self.mesh.ncells):
            betadiv=0.0
            for ii in range(3):
                ie =  self.mesh.edgesOfCell[it,ii]
                if it ==  self.mesh.cellsOfEdge[ie,0]:
                    betadiv += self.betan[ie]
                else:
                    betadiv -= self.betan[ie]
            if abs(betadiv)>1e-8:
                raise ValueError("problem in betan: betadiv=%g" %(betadiv))
        if self.finaltime:
            self.nitertime = int(np.sqrt(self.mesh.ncells)/self.cfl)
            self.dt = self.finaltime/self.nitertime
            self.b = np.zeros(self.mesh.nnodes)
            self.du = np.zeros(self.mesh.nnodes)
            self.r = np.zeros(self.mesh.nnodes)
            self.umoinsun = np.zeros(self.mesh.nnodes)
            self.umoinsdeux = np.zeros(self.mesh.nnodes)
        else:
            self.dt = None

    def defineProblem(self):
        problemsplit = self.problem.split('_')
        if problemsplit[0] == 'Analytic':
            assert self.dirichlet is None
            assert self.rhs is None
            assert self.solexact is None
            if self.alpha != 0.0:
               print(3*'*** alpha not zero: linear function is not reproduced exactly! ')
            if self.beta is None:
                self.beta = lambda x, y: (0.4, 0.6)
            if problemsplit[1] == 'Linear':
                self.solexact = AnalyticalSolution('0.3 * x + 0.7 * y')
            elif problemsplit[1] == 'Quadratic':
                self.solexact = AnalyticalSolution('x*x+2*y*y')
            elif problemsplit[1] == 'Exponential':
                self.solexact = AnalyticalSolution('exp(x-0.7*y)')
            elif problemsplit[1] == 'Sinus':
                self.solexact = AnalyticalSolution('sin(x+0.2*y*y)')
            elif problemsplit[1] == 'Constant':
                self.solexact = AnalyticalSolution('7.0')
            else:
                raise ValueError("unknown analytic solution: '%s'" %(problemsplit[1]))
            self.dirichlet = self.solexact
            self.dontcomputeBcells = False
            self.errorformula = 'nodes'
        elif problemsplit[0] == 'Ramp':
            assert self.dirichlet is None
            assert self.rhs is None
            assert self.solexact is None
            assert self.beta is None
            if self.alpha != 0.0:
                print('***WARNING alpha=',alpha)
            self.beta = lambda x, y: (0.1, 0.6)
            def dir(x, y):
                if x > 0: return 1.0
                if x < 0: return -1.0
            self.dirichlet = dir
            def solex(x, y):
                r = 0.6*x-0.1*y -0.1
                return np.piecewise(r, [r <= 0., r > 0.], [-1, 1])
            self.solexact = solex
            self.dontcomputeBcells = True
            self.errorformula = 'cell'
        elif problemsplit[0] == 'RotStat':
            assert self.dirichlet is None
            assert self.rhs is None
            assert self.solexact is None
            assert self.beta is None
            assert self.alpha == 0
            self.beta = lambda x, y: (1.0+y, -1.0-x)
            def dir(x, y):
                if np.abs(y) <= 0.5: return 1.0
                else: return 0.0
            self.dirichlet = dir
            def solex(x, y):
                r = np.sqrt((x+1.0)**2+(y+1.0)**2)
                return np.piecewise(r, [ (0<=r) & (r < 0.5),  (0.5<=r) & (r < 1.5)], [0 , 1, 0])
            self.solexact = solex
            self.dontcomputeBcells = True
            self.errorformula = 'cell'
        assert self.beta
        self.beta = np.vectorize(self.beta)
        assert self.dirichlet
    def computeBcells(self, rhs=None):
        if self.dontcomputeBcells:
            return None
        bcells = None
        solexact = self.solexact
        xmid, ymid =  self.mesh.centersx,  self.mesh.centersy
        if solexact:
            assert rhs is None
            u = solexact(xmid, ymid)
            gradu = np.stack((solexact.x(xmid, ymid), solexact.y(xmid, ymid)), axis=-1)
            beta = np.stack(self.beta(xmid, ymid),axis=-1)
            # print 'gradu', gradu.shape
            # print 'beta', beta.shape
            # print  'dot', np.inner(gradu[:], beta[:]).shape
            # stop
            # bcells = np.dot(gradu, self.beta(xmid, ymid)) *  self.mesh.area[:]
            bcells = np.zeros(gradu.shape[0])
            for i in range(gradu.shape[0]):
                bcells[i] = (self.alpha*u[i] + np.dot(gradu[i],beta[i]) )* self.mesh.area[i]
        elif rhs:
            assert solexact is None
            bcells = rhs(xmid, ymid) *  self.mesh.area[:]
        return bcells
    def postProcess(self, u, timer, nit=True):
        info = {}
        info['timer'] = self.timer
        if nit: info['nit'] = nit
        errors, point_data, cell_data = self.computeError(self.solexact, u)
        info['error'] = errors
        if self.vtkbeta:
            beta = self.beta( self.mesh.x,  self.mesh.y)
            point_data['betax'] = beta[0]
            point_data['betay'] = beta[1]
        return point_data, cell_data, info
    def overundershoots(self, u, ue):
        assert u.shape[0] == ue.shape[0]
        min = np.min(ue)
        max = np.max(ue)
        imin, = np.where(u < min)
        imax, = np.where(u > max)
        maxundershoot=maxovershoot=0.0
        if imin.shape[0]:
            maxundershoot = min - np.min(u[imin])
        if imax.shape[0]:
            maxovershoot = np.max(u[imax]) - max
        return {'undershoot': maxundershoot, 'overshoot': maxovershoot}
    def computeErrorCellvector(self, solexact, u, checkmonotone=True):
        assert u.shape[0] == self.mesh.ncells
        solvec = solexact(self.mesh.centersx, self.mesh.centersy)
        err = 0.0
        if self.errorformula == 'cell':
            errname = 'l1'
            for ic in range(self.mesh.ncells):
                err += abs(solvec[ic] - u[ic]) * self.mesh.area[ic]
        elif self.errorformula == 'nodes':
            errname = 'l2'
            ue = solexact(self.mesh.x, self.mesh.y)
            elem = self.mesh.triangles
            for ic in range(self.mesh.ncells):
                for ii in range(3):
                    err += (ue[elem[ic, ii]] - u[ic]) ** 2 * self.mesh.area[ic] / 3.0
            err = np.sqrt(err)
        else:
            raise ValueError("unknown error formula self.errorformula=%s" % self.errorformula)
        errors = {errname: err}
        if checkmonotone:
            errou = self.overundershoots(u, solvec)
            errors.update(errou)
        pointdata = {}
        celldata = {}
        celldata['U'] = u
        celldata['UE'] = solvec
        celldata['E'] = np.abs(solvec-u)
        return errors, pointdata, celldata
    def computeErrorNodevector(self, solexact, u, checkmonotone=True):
        assert u.shape[0] == self.mesh.nnodes
        solvec = solexact(self.mesh.x, self.mesh.y)
        elem = self.mesh.triangles
        err = 0.0
        if self.errorformula == 'cell':
            ue = solexact(self.mesh.centersx, self.mesh.centersy)
            errname = 'l1'
            for ic in range(self.mesh.ncells):
                uh = np.mean(u[elem[ic]])
                # print 'ue', ue[ic], uh
                err += abs(ue[ic] - uh) * self.mesh.area[ic]
        elif self.errorformula == 'nodes':
             errname = 'l2'
             for ic in range(self.mesh.ncells):
                for ii in range(3):
                    err += (solvec[elem[ic, ii]] - u[elem[ic, ii]]) ** 2 * self.mesh.area[ic] / 3.0
             err = np.sqrt(err)
        else:
            raise ValueError("unknown error formula self.errorformula=%s" % self.errorformula)
        errors = {errname: err}
        if checkmonotone:
            errou = self.overundershoots(u, solvec)
            errors.update(errou)
        pointdata = {}
        celldata = {}
        pointdata['U'] = u
        pointdata['UE'] = solvec
        pointdata['E'] = np.abs(solvec-u)
        return errors, pointdata, celldata
    def computeErrorEdgevector(self, solexact, u, checkmonotone=True):
        assert u.shape[0] == self.mesh.nedges
        solvec = solexact(self.mesh.centersx, self.mesh.centersy)
        # print 'u', u
        uc = np.zeros(self.mesh.ncells)
        for ic in range(self.mesh.ncells):
            for ii in range(3):
                ind = self.mesh.edgesOfCell[ic, ii]
                uc[ic] += u[ind]/3.0
                # uc[ic] = np.mean( u[self.mesh.edgesOfCell[ic,:]])
        # uc = np.mean( u[self.mesh.edgesOfCell], axis=1 )
        assert uc.shape[0] == self.mesh.ncells
        err = 0.0
        if self.errorformula == 'nodes' or checkmonotone:
            ue = solexact(self.mesh.edgesx, self.mesh.edgesy)
            # print 'ue', ue
        if self.errorformula == 'cell':
            errname = 'l1'
            for ic in range(self.mesh.ncells):
                err += abs(solvec[ic] - uc[ic]) * self.mesh.area[ic]
        elif self.errorformula == 'nodes':
            errname = 'l2'
            for ic in range(self.mesh.ncells):
                for ii in range(3):
                    ind = self.mesh.edgesOfCell[ic, ii]
                    err += (ue[ind] - u[ind]) ** 2 * self.mesh.area[ic] / 3.0
            err = np.sqrt(err)
        else:
            raise ValueError("unknown error formula self.errorformula=%s" % self.errorformula)
        errors = {errname: err}
        if checkmonotone:
            errou = self.overundershoots(u, ue)
            errors.update(errou)
        pointdata = {}
        celldata = {}
        celldata['U'] = uc
        celldata['UE'] = solvec
        celldata['E'] = np.abs(solvec-uc)
        return errors, pointdata, celldata
    def computeError(self, solexact, u, checkmonotone=True):
        pointdata = {}
        celldata = {}
        if u.shape[0] == self.mesh.ncells:
            return self.computeErrorCellvector(solexact, u, checkmonotone)
        elif u.shape[0] == self.mesh.nnodes:
            return self.computeErrorNodevector(solexact, u, checkmonotone)
        elif u.shape[0] == self.mesh.nedges:
            return self.computeErrorEdgevector(solexact, u, checkmonotone)
        else:
            raise ValueError('solution vector has unknown size %d' %(u.shape[0]))
        return errors, pointdata, celldata
    def checkMMatrix(self, A):
        problem = False
        for i in range(A.indptr.shape[0]-1):
            sum = 0.0
            for pos in range(A.indptr[i], A.indptr[i+1]):
                j = A.indices[pos]
                if  j==i:
                    diag = A.data[pos]
                else:
                    aij = A.data[pos]
                    if aij>1e-16:
                        problem = True
                        print("positive off-diagonal in %d-%d: %g"  %(i, j, aij))
                    sum += A.data[pos]<=0.0
            if diag + sum < 0.0:
                problem = True
                print("summ condition voilated in row %d sum=%g diag=%g" %(i, sum, diag))
        if problem:
            import matplotlib.pyplot as plt
            self.mesh.plot(plt)
            import  sys
            sys.exit(1)
    def solve(self):
        return self.solveNonlinear()
    def solveDynamic(self, x, b, redrate, it):
        if redrate>0.001:
            self.A = self.matrixDynamic(x)
            self.countmatrix += 1
        du =  splinalg.spsolve(self.A, b)
        return du
    def solvedynamic(self, name, meshiter, dirname):
        t = 0.0
        u = self.computeInitialcondition()
        self.umoinsun[:] = u[:]
        point_data = {"u": u}
        filename = "%s%02d_%04d.vtk" % (name, meshiter, 0)
        self.mesh.write(filename=filename, dirname=dirname, point_data=point_data)
        t = 0.0
        # outputfilter = meshiter+2
        outputfilter = max(1, int( (self.nitertime + 1) / (1.0 * self.nvisusteps)))
        nittotal = 0
        self.matrixDynamic(u)
        self.countmatrix = 0
        for iter in range(1, self.nitertime + 1):
            t += self.dt
            self.umoinsdeux[:] = self.umoinsun[:]
            self.umoinsun[:] = u[:]
            t0 = time.clock()
            self.computeRhsDynamic(u)
            t1 = time.clock()
            t2 = time.clock()
            u, res, nit = newton(self.residualDynamic, self.solveDynamic, u, rtol=1e-10, gtol=1e-16, maxiter=20,
                                 checkmaxiter=True, silent=True)
            nittotal += nit
            s = "%d/%d (%d)" % (iter, self.nitertime + 1, nit)
            print(len(s) * '\r', s, end=' ')
            t3 = time.clock()
            self.timer['rhs'] += t1 - t0
            self.timer['matrix'] += t2 - t1
            self.timer['solve'] += t3 - t2
            point_data = {"u": u}
            if iter % outputfilter == 0:
                filename = "%s%02d_%04d.vtk" % (name, meshiter, iter / outputfilter)
                self.mesh.write(filename=filename, dirname=dirname, point_data=point_data)
        print()
        print('self.countmatrix', self.countmatrix)
        info = {}
        info['timer'] = self.timer
        if nit: info['nit'] = nittotal
        errors, point_data, cell_data = self.computeErrorNodevector(np.vectorize(self.solexact), u)
        filename = "%s%02d_finalerror.vtk" % (name, meshiter)
        self.mesh.write(filename=filename, dirname=dirname, point_data=point_data)
        info['error'] = errors
        return point_data, cell_data, info


# ------------------------------------- #

if __name__ == '__main__':
    beta = lambda x, y: (1.0, 2.0)
    trimesh = TriMesh(hnew=0.4)
    # Transport
