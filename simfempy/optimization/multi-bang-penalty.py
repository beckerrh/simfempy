# Created by becker at 2019-05-04
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class MultiBangPenalty():
    def __init__(self, uin):
        print("uin", uin)
        u = np.copy(uin)
        if u[0]!=0:
            self.u0 = u[0]
            u -= u[0]
        else:
            self.u0 = 0
        assert(u[0]==0)
        print("u", u)
        print("np.diff(u)", np.diff(u))
        self.u = u
        self.m = u.shape[0]
        self.penalty = np.vectorize(self._penalty)
    def _penalty(self, tin):
        t = tin-self.u0
        u, m = self.u, self.m
        if t <= u[0]:
            return -u[-1]*t
        for i in range(m-1):
            if t <= u[i+1]:
                return 0.5*( t*(u[i]+u[i+1]) - u[i]*u[i+1])
        return t*u[-1] - 0.5*u[-1]**2


u = np.arange(1, 4)
mbp = MultiBangPenalty(u)
up = np.linspace(-0.2*u[0], 1.2*u[-1], 100)
mp = mbp.penalty(up)
plt.plot(up, mp)
plt.show()