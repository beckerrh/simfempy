import numpy as np
import pyamg
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools
import time

scipysolvers=['gmres','lgmres','gcrotmk','bicgstab','cgs']
strangesolvers=['gmres']

#-------------------------------------------------------------------#
def selectBestSolver(solvers, reduction, b, **kwargs):
    maxiter = kwargs.pop('maxiter', 50)
    verbose = kwargs.pop('verbose', 0)
    tol = kwargs.pop('tol') if not 'tol' in kwargs else 0.1*reduction
    analysis = {}
    for solvername, solver in solvers.items():
        t0 = time.time()
        res = solver.testsolve(b=b, maxiter=maxiter, tol=tol)
        t = time.time() - t0
        monotone = np.all(np.diff(res) < 0)
        if len(res)==1:
            if res[0] > 1e-6: 
                print(f"no convergence in {solvername=} {res=}")
                continue
            iterused = 1
        else:
            rho = np.power(res[-1]/res[0], 1/len(res))
            if not monotone:
                print(f"***VelcoitySolver {solvername} not monotone {rho=}")
                continue
            if rho > 0.8: 
                print(f"***VelcoitySolver {solvername} bad {rho=}")
                continue
            iterused = int(np.log(reduction)/np.log(rho))+1
        treq = t/len(res)*iterused
        analysis[solvername] = (iterused, treq)
    # print(f"{self.analysis=}")
    if verbose:
        for solvername, val in analysis.items():
            print(f"{solvername=} {val=}")
    if len(analysis)==0: raise ValueError('*** no working solver found')
    ibest = np.argmin([v[1] for v in analysis.values()])
    solverbest = list(analysis.keys())[ibest]
    # print(f"{solverbest=}")
    # self.solver = self.solvers[solverbest]
    return solverbest, analysis[solverbest][0]


#=================================================================#
class ScipySpSolve():
    def __init__(self, **kwargs):
        self.matrix = kwargs.pop('matrix')
    def solve(self, b, maxiter, tol, x0=None):
        return splinalg.spsolve(self.matrix, b)
    def testsolve(self, b, maxiter, tol):
        splinalg.spsolve(self.matrix, b)
        return [0]

#=================================================================#
class ScipySolve():
    def __repr__(self):
        return "scipy_"+self.method
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method')
        if self.method in strangesolvers: raise ValueError(f"method '{self.method}' is i strange scipy solver")
        if "matrix" in kwargs:
            self.matvec = kwargs.pop('matrix')
            if not "matvecprec" in kwargs:
                fill_factor = kwargs.pop("fill_factor", 2)
                drop_tol = kwargs.pop("fill_factor", 0.01)
                # spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=0.1, fill_factor=2)
                spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
                self.M = splinalg.LinearOperator(self.matvec.shape, lambda x: spilu.solve(x))
        else:
            # self.matvec = kwargs.pop('matvec')
            if not 'n' in kwargs: raise ValueError(f"need 'n' if no matrix given")
            n = kwargs.get('n')
            self.matvec = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvec'))
        if "matvecprec" in kwargs:
            n = kwargs.get('n')
            self.M = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvecprec'))
        else:
            self.M = None
        self.atol = 1e-14
        disp = kwargs.pop('disp', 0)
        # print(f"**** {disp=}")
        self.args = {"A": self.matvec, "M":self.M, "atol":self.atol}
        self.solver = eval('splinalg.'+self.method)
        name = self.method
        if self.method=='gcrotmk':
            self.args['m'] = kwargs.pop('m', 5)
            self.args['truncate'] = kwargs.pop('truncate', 'smallest')
            self.solver = splinalg.gcrotmk
            name += '_' + str(self.args['m'])
        if 'counter' in kwargs:
            self.counter = tools.iterationcounter.IterationCounter(name=kwargs.pop('counter')+str(self), disp=disp)
            self.args['callback'] = self.counter
    def solve(self, b, maxiter, tol, x0=None):
        if hasattr(self, 'counter'):
            self.counter.reset()
            self.args['callback'] = self.counter
        self.args['b'] = b
        self.args['maxiter'] = maxiter
        self.args['x0'] = x0
        self.args['tol'] = tol
        u, info = self.solver(**self.args)
        # if info: raise ValueError(f"no convergence in {self.method=} {info=}")
        return u
    def testsolve(self, b, maxiter, tol):
        # print(f"{np.linalg.norm(b)=} {maxiter=} {tol=}")
        counter = tools.iterationcounter.IterationCounterWithRes(name=self.method, callback_type='x', disp=0, b=b, A=self.matvec)
        args = self.args.copy()
        args["callback"] = counter
        args['b'] = b
        args['maxiter'] = maxiter
        args['tol'] = tol
        u, info = self.solver(**args)
        # print(f"{counter.res=}")
        return counter.history

#=================================================================#
class Pyamg():
    def __repr__(self):
        return "pyamg_"+self.type+self.smoother+str(self.accel)
    def __init__(self, A, **kwargs):
        self.A = A
        nsmooth = kwargs.pop('nsmooth', 1)
        self.smoother = kwargs.pop('smoother', 'schwarz')
        symmetric = kwargs.pop('symmetric', False)
        self.type = kwargs.pop('type', 'aggregation')
        self.accel = kwargs.pop('accel', None)
        if self.accel == 'none': self.accel=None
        pyamgargs = {'B': pyamg.solver_configuration(A, verb=False)['B'], 'presmoother':self.smoother, 'postsmoother':self.smoother}
        if symmetric:
            smooth = ('energy', {'krylov': 'cg'})
        else:
            smooth = ('energy', {'krylov': 'fgmres'})
            pyamgargs['symmetry'] = 'nonsymmetric'
        # pyamgargs['smooth'] = smooth
        # pyamgargs['coarse_solver'] = 'splu'
        if self.type == 'aggregation':
            self.solver = pyamg.smoothed_aggregation_solver(A, **pyamgargs)
        elif self.type == 'rootnode':
            self.solver = pyamg.rootnode_solver(A, **pyamgargs)
        else:
            raise ValueError(f"unknown {self.type=}")
        disp = kwargs.pop('disp', 0)
        self.solveargs = {'cycle': 'V', 'accel': self.accel}
        if 'counter' in kwargs:
            self.counter = tools.iterationcounter.IterationCounter(name=kwargs.pop('counter')+str(self), disp=disp)
            self.solveargs['callback'] = self.counter
#        cycle : {'V','W','F','AMLI'}

    def testsolve(self, b, maxiter, tol):
        counter = tools.iterationcounter.IterationCounterWithRes(name=self, callback_type='x', disp=0, b=b, A=self.A)
        # counter = tools.iterationcounter.IterationCounter(name=self, disp=0)
        args = self.solveargs.copy()
        args['callback'] = counter
        self.solver.solve(b, maxiter=maxiter, tol=tol, **args)
        return counter.history
    def solve(self, b, maxiter, tol):
        if hasattr(self, 'counter'):
            self.counter.reset()
        return self.solver.solve(b, maxiter=maxiter, tol=tol, **self.solveargs)
