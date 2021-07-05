import numpy as np
import pyamg
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools
import time

scipysolvers=['scipy_gmres','scipy_lgmres','scipy_gcrotmk','scipy_bicgstab','scipy_cgs', 'scipy_cg']
pyamgsolvers=['pyamg_gmres','pyamg_fgmres','pyamg_bicgstab', 'pyamg_cg']
strangesolvers=['gmres']

#-------------------------------------------------------------------#
def _getSolver(args):
    if not isinstance(args, dict): raise ValueError(f"*** args must be a dict")
    if not 'method' in args: raise ValueError(f"*** needs 'method' in args\ngiven: {args}")
    method = args.pop('method')
    matrix = args.pop('matrix', None)
    if method in scipysolvers or method in pyamgsolvers:
        if matrix is not None:
            return ScipySolve(matrix=matrix, method=method, **args)
        else:
            return ScipySolve(method=method, **args)
    elif method == "spsolve":
        return ScipySpSolve(matrix=matrix)
    elif method == "pyamg":
        return Pyamg(matrix, **args)
    else:
        raise ValueError(f"unknwown {method=}")


#-------------------------------------------------------------------#
def getSolver(**kwargs):
    args = kwargs.pop('args', 50)
    if isinstance(args, dict): return _getSolver(args)
    maxiter = args.pop('maxiter', 50)
    verbose = args.pop('verbose', 0)
    reduction = args.pop('reduction', 0.01)
    rtol = args.pop('rtol') if 'rtol' in args else 0.1*reduction
    solvers = {}
    for arg in args:
        solvers[arg] = _getSolver(arg)
        n = solvers[arg].shape[0]
    b = np.random.random(n)
    b /= np.linalg.norm(b)
    analysis = {}
    for solvername, solver in solvers.items():
        t0 = time.time()
        res = solver.testsolve(b=b, maxiter=maxiter, rtol=rtol)
        t = time.time() - t0
        monotone = np.all(np.diff(res) < 0)
        if len(res)==1:
            if res[0] > rtol:
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
    if verbose:
        print(f"{solverbest=}")
    return solvers[solverbest], analysis[solverbest][0]
#=================================================================#
class ScipySpSolve():
    def __init__(self, **kwargs):
        self.matrix = kwargs.pop('matrix')
    def solve(self, b, maxiter=None, rtol=None, x0=None):
        return splinalg.spsolve(self.matrix, b)
    def testsolve(self, b, maxiter, rtol):
        splinalg.spsolve(self.matrix, b)
        return [0]
#=================================================================#
class IterativeSolver():
    def __repr__(self):
        return f"{self.method}_{self.maxiter}_{self.rtol}"
    def __init__(self, **kwargs):
        self.args = {}
        self.atol = kwargs.pop('atol', 1e-14)
        self.rtol = kwargs.pop('rtol', 1e-8)
        self.maxiter = kwargs.pop('maxiter', 100)
        if 'counter' in kwargs:
            disp = kwargs.pop('disp', 0)
            self.counter = tools.iterationcounter.IterationCounter(name=kwargs.pop('counter')+str(self), disp=disp)
            self.args['callback'] = self.counter
    def solve(self, b, maxiter=None, rtol=None, x0=None):
        # print(f"{maxiter=} {self.maxiter=}")
        # print(f"{rtol=} {self.args=}")
        if maxiter is None: maxiter = self.maxiter
        if rtol is None: rtol = self.rtol
        if hasattr(self, 'counter'):
            self.counter.reset()
            self.args['callback'] = self.counter
        self.args['b'] = b
        self.args['maxiter'] = maxiter
        self.args['x0'] = x0
        self.args['tol'] = rtol
        res  = self.solver(**self.args)
        return res[0] if isinstance(res, tuple) else res
    def testsolve(self, b, maxiter, rtol):
        counter = tools.iterationcounter.IterationCounterWithRes(name=str(self), callback_type='x', disp=0, b=b, A=self.matvec)
        args = self.args.copy()
        args['callback'] = counter
        args['maxiter'] = maxiter
        args['tol'] = rtol
        args['b'] = b
        res = self.solver(**args)
        return counter.history
#=================================================================#
class ScipySolve(IterativeSolver):
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method')
        super().__init__(**kwargs)
        # if self.method in strangesolvers: raise ValueError(f"method '{self.method}' is i strange scipy solver")
        if "matrix" in kwargs:
            self.matvec = kwargs.pop('matrix')
            if not "matvecprec" in kwargs:
                fill_factor = kwargs.pop("fill_factor", 2)
                drop_tol = kwargs.pop("fill_factor", 0.01)
                spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
                self.M = splinalg.LinearOperator(self.matvec.shape, lambda x: spilu.solve(x))
        else:
            if not 'n' in kwargs: raise ValueError(f"need 'n' if no matrix given")
            n = kwargs.get('n')
            self.matvec = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvec'))
        if "matvecprec" in kwargs:
            n = kwargs.get('n')
            self.M = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvecprec'))
        else:
            self.M = None
        # self.args = {"A": self.matvec, "M":self.M, "atol":self.atol}
        self.args['A'] = self.matvec
        self.args['M'] = self.M
        if self.method in scipysolvers:
            self.solver = eval('splinalg.'+self.method[6:])
            self.args['atol'] = self.atol
        elif self.method in pyamgsolvers:
            self.solver = eval('pyamg.krylov.' + self.method[6:])
        else:
            raise ValueError("*** unknown {self.method=}")
        name = self.method
        if self.method=='scipy_gcrotmk':
            self.args['m'] = kwargs.pop('m', 5)
            self.args['truncate'] = kwargs.pop('truncate', 'smallest')
            self.solver = splinalg.gcrotmk
            name += '_' + str(self.args['m'])
#=================================================================#
class Pyamg(IterativeSolver):
    def __repr__(self):
        s = super().__repr__()
        return s + f"pyamg_{self.type}_{self.smoother}_{str(self.accel)}"
    def __init__(self, A, **kwargs):
        self.method = 'pyamg'
        self.matvec = A
        nsmooth = kwargs.pop('nsmooth', 1)
        self.smoother = kwargs.pop('smoother', 'schwarz')
        symmetric = kwargs.pop('symmetric', False)
        self.type = kwargs.pop('pyamgtype', 'aggregation')
        self.accel = kwargs.pop('accel', None)
        if self.accel == 'none': self.accel=None
        pyamgargs = {'B': pyamg.solver_configuration(A, verb=False)['B']}
        smoother = (self.smoother, {'sweep': 'symmetric', 'iterations': nsmooth})
        if symmetric:
            smooth = ('energy', {'krylov': 'cg'})
        else:
            smooth = ('energy', {'krylov': 'fgmres'})
            pyamgargs['symmetry'] = 'nonsymmetric'
        pyamgargs['presmoother'] = smoother
        pyamgargs['postsmoother'] = smoother
        # pyamgargs['smooth'] = smooth
        # pyamgargs['coarse_solver'] = 'splu'
        if self.type == 'aggregation':
            self.mlsolver = pyamg.smoothed_aggregation_solver(A, **pyamgargs)
        elif self.type == 'rootnode':
            self.mlsolver = pyamg.rootnode_solver(A, **pyamgargs)
        else:
            raise ValueError(f"unknown {self.type=}")
        self.solver = self.mlsolver.solve
        #        cycle : {'V','W','F','AMLI'}
        super().__init__(**kwargs)
        self.args['cycle'] = 'V'
        self.args['accel'] = self.accel
#=================================================================#
class SaddlePointSystem():
    """
    A -B.T
    B  0
     or
    A -B.T 0
    B  0   M^T
    0  M   0
    """
    def __init__(self, A, B, M=None):
        self.A, self.B = A, B
        self.na, self.nb = A.shape[0], B.shape[0]
        self.nall = self.na + self.nb
        if M is not None:
            self.M = M
            self.nm = M.shape[0]
            self.nall += self.nm
        constr = hasattr(self, 'M')
        self.matvec = self.matvec3 if constr else self.matvec2
    def matvec3(self, x):
        v, p, lam = x[:self.na], x[self.na:self.na+self.nb], x[self.na+self.nb:]
        w = self.A.dot(v) - self.B.T.dot(p)
        q = self.B.dot(v)+ self.M.T.dot(lam)
        return np.hstack([w, q, self.M.dot(p)])
    def matvec2(self, x):
        v, p = x[:self.na], x[self.na:]
        w = self.A.dot(v) - self.B.T.dot(p)
        q = self.B.dot(v)
        return np.hstack([w, q])
    def to_single_matrix(self):
        nullP = sparse.dia_matrix((np.zeros(self.nb), 0), shape=(self.nb, self.nb))
        A1 = sparse.hstack([self.A, -self.B.T])
        A2 = sparse.hstack([self.B, nullP])
        Aall = sparse.vstack([A1, A2])
        if not hasattr(self, 'M'):
            return Aall.tocsr()
        nullV = sparse.coo_matrix((1, self.na)).tocsr()
        ML = sparse.hstack([nullV, self.M])
        Abig = sparse.hstack([Aall, ML.T])
        nullL = sparse.dia_matrix((np.zeros(1), 0), shape=(1, 1))
        Cbig = sparse.hstack([ML, nullL])
        Aall = sparse.vstack([Abig, Cbig])
        return Aall.tocsr()
#=================================================================#
class PressureSolverScale():
    def __repr__(self):
        return f"pressurescale"
    def __init__(self, coeff):
        n = len(coeff)
        self.BP = sparse.diags(1/coeff, offsets=(0), shape=(n,n))
    def solve(self, b):
        return self.BP.dot(b)
#=================================================================#
class SaddlePointPreconditioner():
    """
    """
    def __repr__(self):
        return f"{self.method=}\n{self.SV=}\n{self.SP=}"
    def __init__(self, AS, **kwargs):
        self.AS = AS
        solver_p = kwargs.pop('solver_p', None)
        solver_v = kwargs.pop('solver_v', None)
        if not isinstance(AS, SaddlePointSystem) or not isinstance(solver_p, (list,dict)) or not isinstance(solver_v,(list,dict)):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem), solver_p, solver_v (dicts of arguments ")
        if isinstance(solver_v,dict):
            solver_v['matrix'] = AS.A
        else:
            for s in solver_v:
                s['matrix'] = AS.A
        self.SV = getSolver(args=solver_v)
        type = solver_p['type']
        if type == 'scale':
            self.SP = PressureSolverScale(coeff = solver_p['coeff'])
        elif type =='diag':
            AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
            solver_p['matrix'] = AS.B @ AD @ AS.B.T
            self.SP = getSolver(solver_p)
        elif type == 'schur':
            solver_p['matvec'] = self.schurmatvec()
            self.SP = getSolver(solver_p)
        self.type = type

        constr = hasattr(AS, 'M')
        self.nv = self.AS.na
        self.nvp = self.AS.na + AS.nb
        self.nall = self.nvp
        if constr: self.nall += AS.m
        method = kwargs.pop('method','full')
        self.method = method
        if method == 'diag':
            self.matvecprec = self.pmatvec3_diag if constr else self.pmatvec2_diag
        elif method == 'triup':
            self.matvecprec = self.pmatvec3_triup if constr else self.pmatvec2_triup
        elif method == 'tridown':
            self.matvecprec = self.pmatvec3_tridown if constr else self.pmatvec2_tridown
        elif method == 'full':
            self.matvecprec = self.pmatvec3_full if constr else self.pmatvec2_full
        else:
            raise ValueError(f"*** unknwon {method=}\npossible values: 'diag', 'triup', 'tridown', 'full'")
    def schurmatvec(self, x):
        v = self.AS.B.T.dot(x)
        v2 = self.SV.solve(v)
        return self.AS.B.dot(v2)
    def pmatvec2_diag(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self.SV.solve(v)
        q = self.SP.solve(p)
        return np.hstack([w, q])
    def pmatvec3_diag(self, x):
        v, p, lam = x[:self.nv], x[self.nv:self.nvp], x[self.nvp:]
        w = self.SV.solve(v)
        q = self.SP.solve(p)
        mu = self.MP.solve(lam)
        return np.hstack([w, q, mu])
    def pmatvec2_triup(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(p)
        w = self.SV.solve(v+self.AS.B.T.dot(q))
        return np.hstack([w, q])
    def pmatvec2_tridown(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self.SV.solve(v)
        q = self.SP.solve(p-self.AS.B.dot(w))
        return np.hstack([w, q])
    def pmatvec2_full(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self.SV.solve(v)
        q = self.SP.solve(p-self.AS.B.dot(w))
        h = self.AS.B.T.dot(q)
        w += self.SV.solve(h)
        return np.hstack([w, q])
