import numpy as np
import scipy.optimize
import time


# ----------------------------------------------------------------#
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

    def __init__(self, solver, **kwargs):
        self.solver = solver
        self.reset()
        self.fullhess = True
        self.gradtest = False
        self.hestest = False
        if 'fullhess' in kwargs: self.fullhess = kwargs.pop('fullhess')
        if 'gradtest' in kwargs: self.gradtest = kwargs.pop('gradtest')
        if 'hestest' in kwargs: self.hestest = kwargs.pop('hestest')
        if 'regularize' in kwargs:
            if not 'nmeasure' in kwargs: raise ValueError("If 'regularize' is given, we need 'nmeasure'")
            if not 'param0' in kwargs: raise ValueError("If 'regularize' is given, we need 'param0'")
            self.regularize = kwargs.pop('regularize')
            if self.regularize is not None:
                self.regularize = np.sqrt(self.regularize)
            else:
                self.regularize = 0
            self.param0 = kwargs.pop('param0')
        else:
            self.regularize = 0
        if 'nparam' in kwargs:
            self.nparam = kwargs.pop('nparam')
        else:
            self.nparam = solver.nparam
        if 'nmeasure' in kwargs:
            self.nmeasure = kwargs.pop('nmeasure')
        else:
            self.nmeasure = solver.nmeasure

        self.lsmethods = ['lm', 'trf', 'dogbox']
        self.minmethods = ['Newton-CG', 'trust-ncg', 'dogleg', 'trust-constr', 'SLSQP', 'BFGS', 'L-BFGS-B', 'TNC']
        self.methods = self.lsmethods + self.minmethods
        self.hesmethods = ['Newton-CG', 'trust-ncg', 'dogleg', 'trust-constr']
        self.boundmethods = ['trf', 'dogbox', 'L-BFGS-B', 'SLSQP', 'TNC']
        if not hasattr(solver, "computeM"):
            print("*** solver does not have 'computeM', setting 'fullhess=False'")
            self.fullhess = False

    # ------------------------------------------------------------------------
    def computeRes(self, param):
        self.r, self.u = self.solver.computeRes(param, self.u)
        if not hasattr(self, 'data0'):
            raise ValueError("Please set data0 !")
        self.r -= self.data0
        if self.regularize:
            self.r = np.append(self.r, self.regularize * (param - self.param0))
        return self.r

    # ------------------------------------------------------------------------
    def computeDRes(self, param):
        self.dr, self.du = self.solver.computeDRes(param, self.u, self.du)
        if self.regularize:
            self.dr = np.append(self.dr, self.regularize * np.eye(self.nparam), axis=0)
        return self.dr

    # ------------------------------------------------------------------------
    def computeJ(self, param):
        return 0.5 * np.linalg.norm(self.computeRes(param)) ** 2

    # ------------------------------------------------------------------------
    def testcomputeDRes(self, param, r, dr, u):
        eps = 1e-6
        if not dr.shape[0] == r.shape[0]:
            raise ValueError("wrong dimensions r.shape={} dr.shape={}".format(r.shape, dr.shape))
        for i in range(param.shape[0]):
            parameps = param.copy()
            parameps[i] += eps
            rp, up = self.solver.computeRes(parameps, u)
            parameps[i] -= 2 * eps
            rm, um = self.solver.computeRes(parameps, u)
            r2 = (rp - rm) / (2 * eps)
            if not np.allclose(dr[:self.nmeasure, i], r2):
                raise ValueError(
                    "problem in computeDRes:\ndr:\n{}\ndr(diff)\n{}\nparam={}\nrp={}\nrm={}".format(dr[:, i], r2, param,
                                                                                                    rp, rm))
            else:
                print(end='#')

    # ------------------------------------------------------------------------
    def computeDJ(self, param, computeRes=True):
        if self.r is None or computeRes:
            self.r = self.computeRes(param)
        self.dr, self.du = self.solver.computeDRes(param, self.u, self.du)
        # print("self.regularize", self.regularize)
        # print("self.r", self.r)
        # print("self.dr", self.dr.shape)
        if self.regularize:
            self.dr = np.append(self.dr, self.regularize * np.eye(self.nparam), axis=0)
        # print("self.dr", self.dr.shape)
        # print("grad=",np.dot(self.r, self.dr))
        if not self.gradtest:
            return np.dot(self.r, self.dr)
        self.testcomputeDRes(param, self.r, self.dr, self.u)

        if hasattr(self.solver, 'computeDResAdjW'):
            dr2 = self.solver.computeDResAdjW(param, self.u)
            if self.regularize: dr2 = np.append(dr2, self.regularize * np.eye(self.nparam), axis=0)
            if not np.allclose(self.dr, dr2):
                raise ValueError(
                    "dr('computeDRes') =\n{}\nbut dr2('computeDResAdjW') =\n{}".format(self.dr[:self.nmeasure],
                                                                                       dr2[:self.nmeasure]))
            # print("r", r.shape, "dr", dr.shape, "np.dot(r,dr)", np.dot(r,dr))

        if hasattr(self.solver, 'computeDResAdjW'):
            grad = np.dot(self.r, self.dr)
            grad2, self.z = self.solver.computeDResAdj(param, self.r[:self.nmeasure], self.u, self.z)
            if self.regularize:
                grad2 += self.regularize * self.regularize * (param - self.param0)
            if not np.allclose(grad, grad2):
                raise ValueError("different gradients\ndirect:  = {}\nadjoint: = {}".format(grad, grad2))
        return np.dot(self.r, self.dr)

    # ------------------------------------------------------------------------
    def testHessian(self, param, gn, M):
        H = gn + M
        H2 = np.zeros_like(H)
        eps = 1e-6
        for i in range(self.nparam):
            parameps = param.copy()
            parameps[i] += eps
            gradp = self.computeDJ(parameps, computeRes=True)
            parameps[i] -= 2 * eps
            gradm = self.computeDJ(parameps, computeRes=True)
            H2[i] = (gradp - gradm) / (2 * eps)
        if np.allclose(H, H2):
            print(end='@')
            return
        print("HD", H - H2)
        raise ValueError(
            "problem in testHessian:\nH:\n{}\nH(diff)\n{}\nparam={}\ngn={}\nM={}".format(H, H2, param, gn, M))

    # ------------------------------------------------------------------------
    def computeDDJ(self, param):
        if self.dr is None:
            self.dr, self.du = self.solver.computeDRes(param)
        gn = np.dot(self.dr.T, self.dr)
        if not self.fullhess:
            # print("Hes=", np.linalg.eigvals(gn))
            if self.hestest:
                self.testHessian(param, gn, np.zeros_like(gn))
            return gn

        self.z = self.solver.computeAdj(param, self.r[:self.nmeasure], self.u, self.z)
        M = self.solver.computeM(param, self.du, self.z)
        # print("gn", np.linalg.eigvals(gn), "M", np.linalg.eigvals(M))
        if not self.hestest:
            return gn + M
        self.testHessian(param, gn, M)
        return gn + M

    # ------------------------------------------------------------------------
    def create_data(self, refparam, percrandom=0, plot=False, printdata=False):
        nmeasures = self.nmeasure
        refdata, self.u = self.solver.computeRes(refparam)
        if refdata.reshape(-1).shape[0] != nmeasures:
            raise ValueError("wrong dim: nmeasures={} but refdata={}".format(nmeasures, refdata))
        if plot: self.solver.plot()
        perturbeddata = refdata * (1 + 0.5 * percrandom * (np.random.rand(nmeasures) - 2))
        if printdata:
            print("refparam", refparam)
            print("refdata", refdata)
            print("perturbeddata", perturbeddata)
        self.data0 = perturbeddata

    # ------------------------------------------------------------------------
    def minimize(self, x0, method, bounds=None, verbose=0, plot=False):
        self.reset()
        if not method in self.boundmethods: bounds = None
        # print("x0", x0, "method", method)
        hascost = True
        t0 = time.time()
        #         if bounds is None or method == 'lm': bounds = (-np.inf, np.inf)
        # scipy.optimize.show_options("minimize",method=method, disp=True)
        if method in self.lsmethods:
            if bounds is None: bounds = (-np.inf, np.inf)
            else: bounds = (bounds.lb, bounds.ub)
            print("bounds", bounds)
            info = scipy.optimize.least_squares(self.computeRes, jac=self.computeDRes, x0=x0,
                                                method=method, bounds=bounds, verbose=verbose)
        elif method in self.minmethods:
            hascost = False
            if method in self.hesmethods:
                hess = self.computeDDJ
            else:
                hess = None
            options = None
            # method = 'trust-constr'
            callback = None
            if method == 'Newton-CG':
                tol = None
                if verbose == 2:
                    options = {'xtol': 1e-16, 'disp':True}
                    def printiter(x0): print("x0", x0)
                    callback = printiter
            else:
                tol = 1e-12
            # if bounds and len(bounds) == 2: bounds = [bounds for l in range(len(x0))]
            info = scipy.optimize.minimize(self.computeJ, x0=x0, jac=self.computeDJ, hess=hess,
                                           method=method, bounds=bounds, tol=tol, callback=callback,
                                           options=options)
        else:
            raise NotImplementedError("unknown method '{}' known are {}".format(method, ','.join(self.methods)))
        dt = time.time() - t0
        # if method == 'trust-ncg': print(info)
        # print("info", info)
        if hascost:
            cost = info.cost
        else:
            cost = info.fun
        if hasattr(info, 'nhev'):
            nhev = info.nhev
        else:
            nhev = 0
        if hasattr(info, 'njev'):
            njev = info.njev
        else:
            njev = 0
        nfev = info.nfev
        if not info.success:
            print(10 * "@" + " no convergence!")
            nfev, njev, nhev = -1, -1, -1
        if hasattr(self.solver, 'param2x'):
            xf = self.solver.param2x(info.x)
        else:
            xf = info.x
        x = np.array2string(xf, formatter={'float_kind': lambda x: "%11.4e" % x})
        print(
            "{:^14s} x = {} J={:10.2e} nf={:4d} nj={:4d} nh={:4d} {:10.2f} s".format(method, x, cost, nfev, njev, nhev, dt))
        if plot:
            self.solver.plot(suptitle="{}".format(method))
        return xf, cost, info.nfev, njev, nhev, dt

    # ------------------------------------------------------------------------
    def testmethods(self, x0, methods, bounds=None, plot=False, verbose=0):
        values = {"J": [], "nf": [], "ng": [], "nh": [], "s": []}
        xall = np.empty(shape=(len(methods), self.nparam))
        valformat = {"J": "10.2e", "nf": "3d", "ng": "3d", "nh": "3d", "s": "6.1f"}
        for i, method in enumerate(methods):
            x, cost, nfev, njev, nhev, dt = self.minimize(x0=x0, method=method, bounds=bounds, plot=plot,
                                                          verbose=verbose)
            values["J"].append(cost)
            values["nf"].append(nfev)
            values["ng"].append(njev)
            values["nh"].append(nhev)
            values["s"].append(dt)
            xall[i] = x
        return values, valformat, xall
