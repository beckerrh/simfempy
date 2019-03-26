# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import time
import copy
import numpy as np
import scipy.sparse.linalg as splinalg
import scipy.optimize as optimize
import scipy.sparse as sparse

import simfempy.tools.analyticalsolution
import simfempy.tools.timer
import simfempy.tools.iterationcounter
import simfempy.applications.problemdata

# https://github.com/bfroehle/pymumps
#from mumps import DMumpsContext

#=================================================================#
class Solver(object):
    def generatePoblemData(self, exactsolution, bdrycond, postproc=None, random=True):
        problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
        problemdata.ncomp = self.ncomp
        problemdata.solexact = self.defineAnalyticalSolution(exactsolution=exactsolution, random=random)
        problemdata.rhs = self.defineRhsAnalyticalSolution(problemdata.solexact)
        if isinstance(bdrycond, (list, tuple)):
            if len(bdrycond) != self.ncomp: raise ValueError("length of bdrycond ({}) has to equal ncomp({})".format(len(bdrycond),self.ncomp))
            for color in self.mesh.bdrylabels:
                for icomp,bcs in enumerate(problemdata.bdrycond):
                    if bcs.type[color] in ["Dirichlet","Robin"]:
                        bcs.fct[color] = problemdata.solexact[icomp]
                    else:
                        bcs.fct[color] = eval("self.define{}AnalyticalSolution_{:d}(problemdata.solexact)".format(bcs.type[color],icomp))
        else:
            if self.ncomp>1:
                def _solexactdir(x, y, z):
                    return [problemdata.solexact[icomp](x, y, z) for icomp in range(self.ncomp)]
            else:
                def _solexactdir(x, y, z):
                    return problemdata.solexact(x, y, z)
            types = set(problemdata.bdrycond.types())
            types.discard("Dirichlet")
            types.discard("Robin")
            types.add("Neumann")
            problemdata.bdrycond.fctexact = {}
            for type in types:
                cmd = "self.define{}AnalyticalSolution(problemdata.solexact)".format(type)
                problemdata.bdrycond.fctexact[type] = eval(cmd)

            for color in self.mesh.bdrylabels:
                if problemdata.bdrycond.type[color] in ["Dirichlet","Robin"]:
                # if problemdata.bdrycond.type[color] == "Dirichlet":
                        problemdata.bdrycond.fct[color] = _solexactdir
                else:
                    # if color in problemdata.bdrycond.param:
                    #     cmd = "self.define{}AnalyticalSolution(problemdata.solexact,{})".format(bdrycond.type[color],bdrycond.param[color])
                    # else:
                    #     cmd = "self.define{}AnalyticalSolution(problemdata.solexact)".format(bdrycond.type[color])
                    # problemdata.bdrycond.fct[color] = eval(cmd)
                    problemdata.bdrycond.fct[color] = problemdata.bdrycond.fctexact[bdrycond.type[color]]
        return problemdata

    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        return simfempy.tools.analyticalsolution.analyticalSolution(exactsolution, dim, self.ncomp, random)

    def __init__(self, **kwargs):
        self.ncomp = 1
        if 'ncomp' in kwargs: self.ncomp = kwargs.pop('ncomp')
        if 'geometry' in kwargs:
            geometry = kwargs.pop('geometry')
            self.mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=1)
            showmesh = True
            if 'showmesh' in kwargs: showmesh = kwargs.pop('showmesh')
            if showmesh:
                self.mesh.plotWithBoundaries()
            return
        self.problemdata = copy.deepcopy(kwargs.pop('problemdata'))
        self.ncomp = self.problemdata.ncomp

        # temporary
        # self.bdrycond = self.problemdata.bdrycond
        # self.postproc = self.problemdata.postproc
        # self.problemdata.rhs = self.problemdata.rhs
        # temporary

        self.linearsolvers=[]
        self.linearsolvers.append('umf')
        self.linearsolvers.append('lgmres')
        # self.linearsolvers.append('bicgstab')
        try:
            import pyamg
            self.linearsolvers.append('pyamg')
        except: pass
        self.linearsolver = 'umf'
        self.timer = simfempy.tools.timer.Timer(verbose=0)

    def setMesh(self, mesh):
        self.mesh = mesh
        self.timer.reset()

    def solveLinear(self):
        self.timer.add('init')
        A = self.matrix()
        self.timer.add('matrix')
        b,u = self.computeRhs()
        self.timer.add('rhs')
        # A,b,u = self.boundary(A, b, u)
        # self.timer.add('boundary')
        u, niter = self.linearSolver(A, b, u, solver=self.linearsolver)
        self.timer.add('solve')
        point_data, cell_data, info = self.postProcess(u)
        self.timer.add('postp')
        info['timer'] = self.timer.data
        info['iter'] = {'lin':niter}
        return point_data, cell_data, info

    def linearSolver(self, A, b, u=None, solver = None, verbose=1):
        if solver is None: solver = self.linearsolver
        if not hasattr(self, 'info'): self.info={}
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD'), 1
        # elif solver == 'scipy-umf_mmd':
        #     return splinalg.spsolve(A, b, permc_spec='MMD_ATA')
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            # defaults: drop_tol=0.0001, fill_factor=10
            M2 = splinalg.spilu(A.tocsc(), drop_tol=0.1, fill_factor=3)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver, verbose=verbose)
            args=""
            cmd = "u = splinalg.{}(A, b, M=M, tol=1e-14, callback=counter {})".format(solver,args)
            exec(cmd)
            return u, counter.niter
        elif solver == 'pyamg':
            import pyamg
            res=[]
            # u = pyamg.solve(A=A, b=b, x0=u, tol=1e-14, residuals=res, verb=False)
            B = np.ones((A.shape[0], 1))
            ml = pyamg.smoothed_aggregation_solver(A, B, max_coarse=10)
            u= ml.solve(b=b, x0=u, tol=1e-12, residuals=res, accel='gmres')
            if(verbose): print('niter ({}) {:4d} ({:7.1e})'.format(solver, len(res),res[-1]/res[0]))
            return u, len(res)
        else:
            raise NotImplementedError("unknown solve '{}'".format(solver))

    def residual(self, u):
        self.du[:]=0.0
        return self.form(self.du, u)- self.b

    def solvefornewton(self, x, b, redrate, iter):
        self.A = self.matrix(x)
        return splinalg.spsolve(self.A, b)
        import pyamg
        x = pyamg.solve(self.A, b, verb=True)
        return x

    def newtonresidual(self, u):
        self.du = self.residual(u)
        # self.A = self.matrix(u)
        return splinalg.spsolve(self.A, self.du)
    def solvefornewtonresidual(self, x, b, redrate, iter):
        x = b
        return x

    def solveNonlinear(self, u=None, rtol=1e-10, gtol=1e-16, maxiter=100, checkmaxiter=True):
        t0 = time.time()
        self.b = self.computeRhs()
        if u is None:
            u = np.zeros_like(self.b)
        else:
            assert u.shape == self.b.shape
        self.du = np.zeros_like(self.b)
        t1 = time.time()
        self.A = self.matrix(u)
        t2 = time.time()

        method = 'broyden2'
        method = 'anderson'
        method = 'krylov'
        method = 'df-sane'
        # method = 'ownnewton'
        # method = 'ownnewtonls'
        if method == 'ownnewton':
            u,res,nit = newton(self.residual, self.solvefornewton, u, rtol=1e-10, gtol=1e-14, maxiter=200)
        elif method == 'ownnewtonls':
            u,res,nit = newton(self.newtonresidual, self.solvefornewtonresidual, u, rtol=1e-10, gtol=1e-14, maxiter=200)
        else:
            self.A = self.matrix(u)
            sol = optimize.root(self.newtonresidual, u, method=method)
            u = sol.x
            nit = sol.nit

        # try:
        #     u,res,nit = newton(self.residual, solve, u, rtol=1e-10, gtol=1e-14)
        # except:
        #     nit = -1
        # print 'nit=', nit
        t3 = time.time()
        pp = self.postProcess(u)
        t4 = time.time()
        self.timer['rhs'] = t1-t0
        self.timer['matrix'] = t2-t1
        self.timer['solve'] = t3-t2
        self.timer['postproc'] = t4-t3
        return pp


# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError("unit test to be written")