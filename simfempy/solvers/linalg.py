import numpy as np
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
        assert matrix is not None
        return ScipySpSolve(matrix=matrix)
    elif method == "pyamg":
        assert matrix is not None
        return Pyamg(matrix, **args)
    else:
        raise ValueError(f"unknwown {method=}")


#-------------------------------------------------------------------#
def getSolver(**kwargs):
    """
    :param kwargs: if args is dict build the correspong solver
    otherwise if args is list, choose the best solver in the list
    :return:
    """
    args = kwargs.pop('args', 50)
    if isinstance(args, dict):
        if len(kwargs): raise ValueError(f"*** unused keys {kwargs}")
        return _getSolver(args)
    assert isinstance(args, list)
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
        if "prec" in kwargs:
            self.M = kwargs.pop("prec")
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
        try:
            import pyamg
        except:
            raise ImportError(f"*** pyamg not found ***")
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
        self.constr = hasattr(self, 'M')
        self.matvec = self.matvec3 if self.constr else self.matvec2
    def scaleAb(self, b):
        assert not self.constr
        DA = self.A.diagonal()
        assert np.all(DA>0)
        AD = sparse.diags(1/DA, offsets=(0), shape=self.A.shape)
        DS = (self.B@AD@self.B.T).diagonal()
        print(f"{DA.max()=} {DA.min()=} {DS.max()=} {DS.min()=}")
        assert np.all(DS>0)
        na = self.A.shape[0]
        nb = self.B.shape[0]
        # DA = np.ones(na)
        # DS = np.ones(nb)
        # SD = sparse.diags(DS, offsets=(0), shape=(nb,nb))
        self.vs = sparse.diags(np.power(DA, -0.5), offsets=(0), shape=self.A.shape)
        # self.ps = sparse.identity(self.A.shape[0])
        self.ps = sparse.diags(np.power(DS, -0.5), offsets=(0), shape=(nb,nb))
        # self.ps = sparse.identity(nb)
        # self.vsi = sparse.diags(np.power(DA, 0.5), offsets=(0), shape=self.A.shape)
        # self.psi = sparse.diags(np.power(DS, 0.5), offsets=(0), shape=(nb,nb))
        self.A = self.vs@self.A@self.vs
        self.B = self.ps@self.B@self.vs
        bv, bp = b[:self.na], b[self.na:]
        b[:self.na] = self.vs@bv
        b[self.na:] = self.ps@bp
    def scaleu(self, u):
        v, p = u[:self.na], u[self.na:]
        u[:self.na] = self.vs@v
        u[self.na:] = self.ps@p
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
        s =  f"{self.method=}\n{self.type=}"
        if hasattr(self,'SV'): s += f"\n{self.SV=}"
        if hasattr(self,'SP'): s += f"\n{self.SP=}"
        return s
    def __init__(self, AS, **kwargs):
        self.AS = AS
        method = kwargs.pop('method','full')
        self.method = method
        solver_p = kwargs.pop('solver_p', None)
        solver_v = kwargs.pop('solver_v', None)
        # print(f"{method=} {solver_p=}")
        constr = hasattr(AS, 'M')
        self.nv = self.AS.na
        self.nvp = self.AS.na + AS.nb
        self.nall = self.nvp
        if constr: self.nall += AS.m
        if method == 'diag':
            self.matvecprec = self.pmatvec3_diag if constr else self.pmatvec2_diag
        elif method == 'triup':
            self.matvecprec = self.pmatvec3_triup if constr else self.pmatvec2_triup
        elif method == 'tridown':
            self.matvecprec = self.pmatvec3_tridown if constr else self.pmatvec2_tridown
        elif method == 'full':
            self.matvecprec = self.pmatvec3_full if constr else self.pmatvec2_full
        elif method[:3] == 'hss':
            ms = method.split('_')
            if len(ms) != 2: raise ValueError(f"*** needs 'hass_alpha'")
            self.alpha = float(ms[1])
            # solver_p['type'] = f"diag_{self.alpha**2}"
            # solver_p['method'] = f"pyamg"
            self.matvecprec = self.pmatvec3_hss if constr else self.pmatvec2_hss
        else:
            raise ValueError(f"*** unknwon {method=}\npossible values: 'diag', 'triup', 'tridown', 'full'")
        if not isinstance(AS, SaddlePointSystem) or not isinstance(solver_p, (list,dict)) or not isinstance(solver_v,(list,dict)):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem), solver_p, solver_v (dicts of arguments ")
        if isinstance(solver_v,dict):
            alpha = self.alpha if hasattr(self,'alpha') else 0
            # solver_v['matrix'] = AS.A + alpha*sparse.identity(AS.A.shape[0])
            solver_v['matrix'] = AS.A
        else:
            for s in solver_v:
                s['matrix'] = AS.A
        self.SV = getSolver(args=solver_v)
        self.type = solver_p['type']
        if self.type == 'scale':
            self.SP = PressureSolverScale(coeff = solver_p['coeff'])
            return
        solver_p['counter'] = '\tP '
        if self.type[:4] =='diag':
            ts = self.type.split('_')
            if len(ts)>1:
                alpha = float(ts[1])
                solver_p['matrix'] = AS.B@ AS.B.T + alpha*sparse.identity(AS.B.shape[0])
            else:
                AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
                solver_p['matrix'] = AS.B @ AD @ AS.B.T
        elif self.type[:5] == 'schur':
            ts = self.type.split('|')
            if len(ts)>1:
                prec = ts[1]
                if prec == 'diag':
                    AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
                    args = {'method':'pyamg', 'maxiter':1}
                    args['matrix'] = AS.B @ AD @ AS.B.T
                    solver_p['prec'] = getSolver(args=args)
                elif prec == 'scale':
                    AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
                    solver_p['prec'] = PressureSolverScale(coeff = AD.diagonal())
                else:
                    raise ValueError(f"unknwon {prec=} {solver_p=}")
            solver_p['matvec'] = self.schurmatvec
            solver_p['n'] = AS.B.shape[0]
            # print(f"## {solver_p=}")
        else:
            raise ValueError(f"*** unknown {self.type=}")
        self.SP = getSolver(args=solver_p)

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
    def pmatvec2_hss(self, x):
        alpha = 0.1
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(p-1/self.alpha*self.AS.B.dot(v))
        w = self.SV.solve(1/self.alpha*v + self.AS.B.T.dot(q))
        return np.hstack([w, q])
