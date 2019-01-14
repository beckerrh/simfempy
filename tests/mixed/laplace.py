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
from fempy import solvers
import fempy.tools.analyticalsolution
import fempy.fems.femrt0

# class Laplace(RaviartThomas):
class Laplace(solvers.solver.NewtonSolver):
    """
    Fct de base de RT0 = sigma * 0.5 * |S|/|K| (x-x_N)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dirichlet = None
        self.rhs = None
        self.solexact = None
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        self.femv = fempy.fems.femrt0.FemRT0()

    def setMesh(self, mesh):
        # super().setMesh(mesh)
        self.mesh = mesh
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
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        info = {}
        cell_data = {'p': u[nfaces:]}
        vc = self.femv.toCell(u[:nfaces])
        for i in range(dim):
            cell_data['v{:1d}'.format(i)] = vc[i::dim]
        point_data = {}
        if self.solexact:
            err, pe, vexx = self.computeError(self.solexact, u[nfaces:], vc)
            cell_data['perr'] = np.abs(pe - u[nfaces:])
            for i in range(dim):
                cell_data['verrx{:1d}'.format(i)] = np.abs(vexx[i] - vc[i::dim])
            # cell_data['verry'] = np.abs(vey - vc[1])
            info['error'] = err
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        return point_data, cell_data, info

    def computeError(self, solexact, p, vc):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        errors = {}
        xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
        pex = solexact(xc, yc, zc)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.dV))
        errv = 0
        vexx=[]
        for i in range(dim):
            solxi = solexact.d(i, xc, yc, zc)
            errv += np.sum( (solxi-vc[i::dim])**2* self.mesh.dV)
            vexx.append(solxi)
        errv = np.sqrt(errv)
        errors['pcL2'] = errp
        errors['vcL2'] = errv
        return errors, pex, vexx

    def computeRhs(self):
        xf, yf, zf = self.femv.pointsf[:, 0], self.femv.pointsf[:, 1], self.femv.pointsf[:, 2]
        xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
        solexact, dirichlet, rhs = self.solexact, self.dirichlet, self.rhs
        if solexact:
            assert rhs is None
            bcells = (solexact.xx(xc, yc, zc) + solexact.yy(xc, yc, zc))* self.mesh.dV
        elif rhs:
            assert solexact is None
            bcells = rhs(xc, yc, zc) *  self.mesh.dV
        bsides = np.zeros(self.mesh.nfaces)
        for color, faces in self.mesh.bdrylabels.items():
            ud = dirichlet(xf[faces], yf[faces], zf[faces])
            bsides[faces] = linalg.norm(self.mesh.normals[faces],axis=1) * ud
        return np.concatenate((bsides, bcells))

    def matrix(self):
        import time
        t0 = time.time()
        A = self.femv.constructMass()
        t1 = time.time()
        B = self.femv.constructDiv()
        t2 = time.time()
        A1 = scipy.sparse.hstack([A,B.T])
        ncells = self.mesh.ncells
        help = np.zeros((ncells))
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncells, ncells))
        A2 = scipy.sparse.hstack([B, help])
        A = scipy.sparse.vstack([A1,A2])
        t3 = time.time()
        A = A.tocsr()
        t4 = time.time()
        print("A", t1-t0, "B", t2-t1, "stack", t3-t2, "csr", t4-t3)
        return A

    def boundary(self, A, b, u):
        return A,b,u

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

    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=True)
    h = [1.0, 0.5, 0.25, 0.125, 0.062, 0.03, 0.015]
    comp.compare(h=h)
