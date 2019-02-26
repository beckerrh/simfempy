import numpy as np
import scipy.optimize
import time

#----------------------------------------------------------------#
class RhsParam(object):
    def __init__(self, param):
        self.param = param
    def __call__(self, x, y, z):
        return self.param


# ----------------------------------------------------------------#
class Optimizer(object):

    def reset(self):
        self.r = None
        self.dr = None
        self.u = None
        self.z = None
        self.du = None

    def __init__(self, solver, nparam=None, nmeasure=None, regularize=None):
        self.solver = solver
        self.reset()
        self.nparam = nparam
        self.nmeasure = nmeasure
        self.regularize = regularize
        if regularize is not None:
            if nmeasure is None: raise ValueError("If 'regularize' is given, we need 'nmeasure'")

    def computeRes(self, param):
        toto = self.solver.computeRes(param, self.u)
        self.r, self.u = self.solver.computeRes(param, self.u)
        return self.r

    def computeDRes(self, param):
        self.dr, self.du = self.solver.computeDRes(param, self.u, self.du)
        return self.dr

    def computeJ(self, param):
        self.r, self.u = self.solver.computeRes(param, self.u)
        return 0.5*np.linalg.norm(self.r)**2

    def computeDJ(self, param):
        if self.r is None:
            self.r, self.u = self.solver.computeRes(param, self.u)
        self.dr, self.du = self.solver.computeDRes(param, self.u, self.du)
        dr2 = self.solver.computeDResAdjW(param, self.u)
        assert np.allclose(self.dr, dr2)
        # print("r", r.shape, "dr", dr.shape, "np.dot(r,dr)", np.dot(r,dr))
        grad = np.dot(self.r, self.dr)
        grad2 = self.solver.computeDResAdj(param, self.r, self.u)
        if not np.allclose(grad, grad2):
            print("self.r", self.r)
            print("self.dr", self.dr)
            raise ValueError("different gradients\ndirect={}\nadjoint={}", grad, grad2)
        return np.dot(self.r, self.dr)

    def computeDDJ(self, param):
        if hasattr(self, 'dr'):
            dr = self.dr
        else:
            dr = self.solver.computeDRes(param)
        # print("r", r.shape, "dr", dr.shape, "np.dot(r,dr)", np.dot(r,dr))
        return np.dot(dr.T, dr)

    def create_data(self, refparam, percrandom=0):
        nmeasures = self.solver.nmeasures
        refdata = self.computeRes(refparam)[:nmeasures]
        perturbeddata = refdata * (1 + 0.5 * percrandom * (np.random.rand(nmeasures) - 2))
        return refdata, perturbeddata

    def minimize(self, x0, method, bounds=None):
        self.reset()
        if bounds is None or method == 'lm': bounds = (-np.inf, np.inf)
        # print("x0", x0, "method", method)
        hascost=True
        hashess = False
        t0 = time.time()
        lsmethods = ['lm', 'trf','dogbox']
        minmethods = ['Newton-CG', 'trust-ncg']
        if method in lsmethods:
            info = scipy.optimize.least_squares(self.computeRes, jac=self.computeDRes, x0=x0, method=method, verbose=0)
        elif method in minmethods:
            hascost = False
            hashess = True
            # method = 'trust-constr'
            info = scipy.optimize.minimize(self.computeJ, x0=x0, jac=self.computeDJ, hess=self.computeDDJ,
                                           method=method)
        else:
            raise NotImplementedError("unknown method '{}' known are {}".format(method,','.join(set.union(set(lsmethods),set(minmethods)))))
        dt = time.time()-t0
        # if method == 'trust-ncg': print(info)
        # print("info", info)
        if not info.success:
            print(10*"@"+" no convergence!")
        if hascost:
            cost = info.cost
        else:
            cost = info.fun
        if hashess:
            nhev = info.nhev
        else:
            nhev = 0
        x = np.array2string(info.x, precision=5, floatmode='fixed')
        print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} nh={:4d} {:10.2f} s".format(method, x, cost, info.nfev, info.njev, nhev, dt))
