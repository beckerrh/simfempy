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
        self.fullhess = True
        self.gradtest = False

    def __init__(self, solver, **kwargs):
        self.solver = solver
        self.reset()
        if 'fullhess' in kwargs: self.fullhess = kwargs.pop('fullhess')
        if 'gradtest' in kwargs: self.gradtest = kwargs.pop('gradtest')
        if 'regularize' in kwargs:
            if not 'nmeasure' in kwargs: raise ValueError("If 'regularize' is given, we need 'nmeasure'")
            if not 'param0' in kwargs: raise ValueError("If 'regularize' is given, we need 'param0'")
            self.regularize = kwargs.pop('regularize')
            if self.regularize is not None: np.sqrt(self.regularize)
            self.nparam = kwargs.pop('nparam')
            self.nmeasure = kwargs.pop('nmeasure')
            self.param0 = kwargs.pop('param0')
        self.lsmethods = ['lm', 'trf','dogbox']
        self.minmethods = ['Newton-CG', 'trust-ncg', 'dogleg']
        self.methods = self.lsmethods +self.minmethods

    def computeRes(self, param):
        self.r, self.u = self.solver.computeRes(param, self.u)
        if self.regularize:
           self.r = np.append(self.r, self.regularize*(param-self.param0))
        return self.r

    def computeDRes(self, param):
        self.dr, self.du = self.solver.computeDRes(param, self.u, self.du)
        if self.regularize:
            self.dr = np.append(self.dr, self.regularize * np.eye(self.nparam),axis=0)
        return self.dr

    def computeJ(self, param):
        return 0.5*np.linalg.norm(self.computeRes(param))**2

    def computeDJ(self, param):
        if self.r is None:
            self.r = self.computeRes(param)
        self.dr, self.du = self.solver.computeDRes(param, self.u, self.du)
        if self.regularize:
            self.dr = np.append(self.dr, self.regularize * np.eye(self.nparam),axis=0)
        if not self.gradtest:
            return np.dot(self.r, self.dr)

        dr2 = self.solver.computeDResAdjW(param, self.u)
        if self.regularize:
            dr2 = np.append(dr2, self.regularize * np.eye(self.nparam),axis=0)
        #     jac[self.nmeasures:, :] = self.regularize * np.eye(self.nparam)

        assert np.allclose(self.dr, dr2)
        # print("r", r.shape, "dr", dr.shape, "np.dot(r,dr)", np.dot(r,dr))
        grad = np.dot(self.r, self.dr)
        grad2, self.z = self.solver.computeDResAdj(param, self.r[:self.nmeasure], self.u, self.z)
        if self.regularize:
            grad2 += self.regularize*self.regularize*(param - self.param0)

        if not np.allclose(grad, grad2):
            raise ValueError("different gradients\ndirect={}\nadjoint={}", grad, grad2)
        return np.dot(self.r, self.dr)

    def computeDDJ(self, param):
        if self.dr is None:
            self.dr, self.du = self.solver.computeDRes(param)
        gn = np.dot(self.dr.T, self.dr)
        if not self.fullhess:
            return gn
        self.z = self.solver.computeAdj(param, self.r[:self.nmeasure], self.u, self.z)
        M = self.solver.computeM(param, self.du, self.z)
        # print("gn", np.linalg.eigvals(gn), "M", np.linalg.eigvals(M))
        return gn+M

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
        if method in self.lsmethods:
            info = scipy.optimize.least_squares(self.computeRes, jac=self.computeDRes, x0=x0, method=method, verbose=0)
        elif method in self.minmethods:
            hascost = False
            hashess = True
            # method = 'trust-constr'
            info = scipy.optimize.minimize(self.computeJ, x0=x0, jac=self.computeDJ, hess=self.computeDDJ, method=method, tol=1e-8)
        else:
            raise NotImplementedError("unknown method '{}' known are {}".format(method,','.join(self.methods)))
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
        x = np.array2string(info.x, formatter={'float_kind':lambda x: "%11.4e" % x})
        print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} nh={:4d} {:10.2f} s".format(method, x, cost, info.nfev, info.njev, nhev, dt))
