# Created by becker at 2019-05-05
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Algebraic():
    def __init__(self):
        """
        A u = b + B q
        c = Cu
        R = c - C A^-1 ( b + B q)
        """
        self.n = 4
        self.nq = 2
        self.nz = 3
        self.A =  np.random.rand(self.n, self.n)
        for i in range(self.n):
            self.A[i,i] += np.sum(self.A[i,:])
        self.B =  np.random.rand(self.n, self.nq)
        self.C =  np.random.rand(self.nz, self.n)

        self.Ainv = np.linalg.inv(self.A)
        self.S = np.dot(self.C, np.dot(self.Ainv, self.B))
        print("nq nz rank(S)", self.nq, self.nz, np.linalg.matrix_rank(self.S))

        self.b =  np.random.rand(self.n)
        self.c =  np.random.rand(self.nz)
        self.q0 = np.random.rand(self.nq)
        print("*q", self.q0)
        u = self.computeState(self.q0)
        self.c = np.dot(self.C,u)
        self.q0[:] = 0

    def computeState(self, q):
        return np.linalg.solve(self.A, self.b + np.dot(self.B, q))

    # def computeRes(self, q, u):
    #     u = np.linalg.solve(self.A, self.b+ np.dot(self.B, q))
    #     return self.c - np.dot(self.C,u)
    # def computeDRes(self, q, u, du):
    #     dr = np.linalg.solve(self.A, self.B)
    #     return dr

    def computeResidual(self, q):
        u = self.computeState(q)
        r = self.c - np.dot(self.C,u)
        return r
    def computeDResidual(self, q):
        return -self.S

if __name__ == '__main__':
    alg = Algebraic()





    import scipy.optimize
    method = 'lm'
    # method = 'trf'
    info = scipy.optimize.least_squares(alg.computeResidual, jac=alg.computeDResidual, x0 = alg.q0, method=method, verbose=2)
    # info = scipy.optimize.least_squares(alg.computeResidual, x0 = alg.q0, method=method, verbose=2)
    print("info", info)
