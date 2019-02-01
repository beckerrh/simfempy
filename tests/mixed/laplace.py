# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse
import scipy.sparse.linalg as splinalg

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(fempypath)
import simfempy.tools.analyticalsolution
import simfempy.fems.femrt0
import simfempy.applications
from simfempy import solvers
from simfempy.tools.timer import Timer


# ------------------------------------- #
class Laplace(solvers.solver.Solver):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.dirichlet = None
        # self.rhs = None
        # self.solexact = None
        # if 'problem' in kwargs:
        #     self.defineProblem(problem=kwargs.pop('problem'))
        print("self.solexact", self.solexact)
        self.femv = simfempy.fems.femrt0.FemRT0()

    def setMesh(self, mesh):
        # super().setMesh(mesh)
        self.mesh = mesh
        self.femv.setMesh(mesh)

    def defineProblem(self, problem):
        self.problemname = problem
        problemsplit = problem.split('_')
        if problemsplit[0] == 'Analytic':
            if problemsplit[1] == 'Constant':
                solexact = simfempy.tools.analyticalsolution.AnalyticalSolution('7')
            elif problemsplit[1] == 'Linear':
                solexact = simfempy.tools.analyticalsolution.AnalyticalSolution('x+2*y')
            elif problemsplit[1] == 'Quadratic':
                solexact = simfempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
            elif problemsplit[1] == 'Hubbel':
                solexact = simfempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
            elif problemsplit[1] == 'Exponential':
                solexact = simfempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
            elif problemsplit[1] == 'Sinus':
                solexact = simfempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
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
        # info['timer'] = self.timer
        # info['runinfo'] = self.runinfo
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
        # solexact, dirichlet, rhs = self.solexact, self.dirichlet, self.rhs
        solexact, rhs = self.solexact, self.rhs
        if solexact:
            assert rhs is None
            bcells = (solexact.xx(xc, yc, zc) + solexact.yy(xc, yc, zc) + solexact.zz(xc, yc, zc))* self.mesh.dV
        elif rhs:
            assert solexact is None
            bcells = rhs(xc, yc, zc) *  self.mesh.dV
        bsides = np.zeros(self.mesh.nfaces)
        for color, faces in self.mesh.bdrylabels.items():
            condition = self.bdrycond.type[color]
            assert condition=="Dirichlet"
            dirichlet = self.bdrycond.fct[color]
            ud = dirichlet(xf[faces], yf[faces], zf[faces])
            bsides[faces] = linalg.norm(self.mesh.normals[faces],axis=1) * ud
        return np.concatenate((bsides, bcells))

    def matrix(self):
        A = self.femv.constructMass()
        B = self.femv.constructDiv()
        return A,B

    def boundary(self, A, b, u):
        return A,b,u

    def _to_single_matrix(self, Ain):
        A, B = Ain
        ncells = self.mesh.ncells
        help = np.zeros((ncells))
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncells, ncells))
        A1 = scipy.sparse.hstack([A, B.T])
        A2 = scipy.sparse.hstack([B, help])
        Aall = scipy.sparse.vstack([A1, A2])
        return Aall.tocsr()

    def linearSolver(self, Ain, bin, u=None, solver = 'umf'):
        solvers = ['umf', 'gmres2']
        solvers = ['gmres2']
        t = Timer("solve")
        for solver in solvers:
            u=self._linearSolver(Ain, bin, u, solver)
            t.add(solver)
        return u

    def _linearSolver(self, Ain, bin, u=None, solver = 'umf'):
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            return splinalg.spsolve(Aall, bin, permc_spec='COLAMD')
        elif solver == 'gmres':
            counter = simfempy.solvers.solver.IterationCounter(name=solver)
            Aall = self._to_single_matrix(Ain)
            u,info = splinalg.lgmres(Aall, bin, callback=counter, inner_m=20, outer_k=4, atol=1e-10)
            if info: raise ValueError("no convergence info={}".format(info))
            return u
        elif solver == 'gmres2':
            nfaces, ncells = self.mesh.nfaces, self.mesh.ncells
            import simfempy.tools.iterationcounter
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver)
            # Aall = self._to_single_matrix(Ain)
            # M2 = splinalg.spilu(Aall, drop_tol=0.2, fill_factor=2)
            # M_x = lambda x: M2.solve(x)
            # M = splinalg.LinearOperator(Aall.shape, M_x)
            A, B = Ain
            A, B = A.tocsr(), B.tocsr()
            D = scipy.sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces,nfaces))
            S = -B*D*B.T
            import pyamg
            config = pyamg.solver_configuration(S, verb=False)
            ml = pyamg.rootnode_solver(S, B=config['B'], smooth='energy')
            # Silu = splinalg.spilu(S)
            # Ailu = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
            def amult(x):
                v,p = x[:nfaces],x[nfaces:]
                return np.hstack( [A.dot(v) + B.T.dot(p), B.dot(v)])
            Amult = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=amult)
            def pmult(x):
                v,p = x[:nfaces],x[nfaces:]
                w = D.dot(v)
                # w = Ailu.solve(v)
                q = ml.solve(p - B.dot(w), maxiter=1, tol=1e-16)
                w = w - D.dot(B.T.dot(q))
                # w = w - Ailu.solve(B.T.dot(q))
                return np.hstack( [w, q] )
            P = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=pmult)
            # u,info = splinalg.gmres(Amult, bin, M=P, callback=counter, atol=1e-10, restart=5)
            u,info = splinalg.lgmres(Amult, bin, M=P, callback=counter, atol=1e-12, tol=1e-12, inner_m=10, outer_k=4)
            if info: raise ValueError("no convergence info={}".format(info))
            return u
        else:
            raise NotImplementedError("solver '{}' ".format(solver))

# ------------------------------------- #
def test_analytic(problem="Analytic_Quadratic", geomname="unitsquare", verbose=2):
    import simfempy.tools.comparerrors
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    if geomname == "unitsquare":
        h = [1.0, 0.5, 0.25, 0.125, 0.062, 0.03, 0.015]
        problem += "_2d"
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrydn'] = "bdrydn:1000,1001"
    elif geomname == "unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06]
        problem += "_3d"
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrydn'] = "bdrydn:100,105"
    methods = {}
    methods['poisson'] = Laplace(problem=problem, bdrycond=bdrycond, postproc=postproc)
    if problem.split('_')[1] == "Linear":
        h = [2, 1, 0.5, 0.25]
    comp = simfempy.tools.comparerrors.CompareErrors(methods, verbose=verbose)
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['pcL2']

# ------------------------------------- #
if __name__ == '__main__':
    test_analytic()