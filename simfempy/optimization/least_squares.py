# Created by becker at 2019-05-06
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------#
class LeastSquares():
    """
    Given a solver class, this class performs different tasks related to least-squares minimization.
    The solver class is required to have two functions for the computation of the residual and its gradient,as in scipy.optimize.least_squares, but with different names:
        - 'res' instead of 'fun' computes the residual vector
        - 'dres' instead of 'jac' computes the residual Jacobian
    The class provides:
    1 - check for consistency of the gradient
    2 - possibly adds a regularization
    3 - provides an interface such that scipy.optimize.minimize can be used
    """
    def __init__(self, solver, **kwargs):
        self.solver = solver

    def checkDerivative(self, x=0):
        eps = 1e-6
        for i in range(param.shape[0]):
            parameps = param.copy()
            parameps[i] += eps
            rp, up = self.solver.computeRes(parameps, u)
            parameps[i] -= 2 * eps
            rm, um = self.solver.computeRes(parameps, u)
            r2 = (rp - rm) / (2 * eps)
            if not np.allclose(dr[:self.nmeasure, i], r2):
                msg = "problem in computeDRes:\ndr:\n{}\ndr(diff)\n{}\nparam={}\nrp={}\nrm={}".format(dr[:, i], r2, param,
                                                                                                    rp, rm)
                raise ValueError(msg)
            else:
                print(end='#')
