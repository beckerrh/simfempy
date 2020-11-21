# -*- coding: utf-8  -*-
"""

"""

import numpy as np
import sympy


#=================================================================#
class AnalyticalSolution():
    """
    computes numpy vectorized functions for the function and its dericatives up to two
    for a given expression, derivatives computed with sympy
    """
    def __repr__(self):
        return f"dim={self.dim} expr={str(self.expr)}"
    def __call__(self, *x):
        return self.fct(*x)
    def __init__(self, expr, dim=3):
        if expr.find('x0') == -1 and expr.find('x1') == -1 and expr.find('x2') == -1:
            expr = expr.replace('x', 'x0')
            expr = expr.replace('y', 'x1')
            expr = expr.replace('z', 'x2')
        if dim==1 and expr.find('x0') == -1:
            expr = expr.replace('x', 'x0')
        self.dim, self.expr = dim, expr
        symbc = ""
        for i in range(dim): symbc += f"x{i},"
        symbc = symbc[:-1]
        s = sympy.symbols(symbc)
        # print(f"{expr=} {symbc=} {s=}")
        self.fct = np.vectorize(sympy.lambdify(symbc,expr))
        self.fct_x = []
        self.fct_xx = []
        for i in range(dim):
            if dim==1: fx = sympy.diff(expr, s)
            else: fx = sympy.diff(expr, s[i])
            self.fct_x.append(np.vectorize(sympy.lambdify(symbc, fx),otypes=[float]))
            self.fct_xx.append([])
            for j in range(dim):
                if dim == 1: fxx = sympy.diff(fx, s)
                else: fxx = sympy.diff(fx, s[j])
                self.fct_xx[i].append(np.vectorize(sympy.lambdify(symbc, fxx),otypes=[float]))
    def d(self, i, *x):
        return self.fct_x[i](*x)
    def x(self, *x):
        return self.fct_x[0](*x)
    def y(self, *x):
        return self.fct_x[1](*x)
    def z(self, *x):
        return self.fct_x[2](*x)
    def dd(self, i, j, *x):
        return self.fct_xx[i][j](*x)
    def xx(self, *x):
        return self.fct_xx[0][0](*x)
    def xy(self, *x):
        return self.fct_xx[0][1](*x)
    def xz(self, *x):
        return self.fct_xx[0][2](*x)
    def yy(self, *x):
        return self.fct_xx[1][1](*x)
    def yx(self, *x):
        return self.fct_xx[1][0](*x)
    def yz(self, *x):
        return self.fct_xx[1][2](*x)
    def zz(self, *x):
        return self.fct_xx[2][2](*x)
    def zx(self, *x):
        return self.fct_xx[2][0](*x)
    def zy(self, *x):
        return self.fct_xx[2][1](*x)


#=================================================================#
# class AnalyticalSolution3d():
#     """
#     computes numpy vectorized functions for the function and its dericatives up to two
#     for a given expression, derivatives computed with sympy
#     """
#     def __init__(self, expr):
#         (x, y, z) = sympy.symbols('x,y,z')
#         self.expr = expr
#         self.fct = np.vectorize(sympy.lambdify('x,y,z',expr))
#         fx = sympy.diff(expr, x)
#         fy = sympy.diff(expr, y)
#         fz = sympy.diff(expr, z)
#         fxx = sympy.diff(fx, x)
#         fxy = sympy.diff(fx, y)
#         fxz = sympy.diff(fx, z)
#         fyy = sympy.diff(fy, y)
#         fyz = sympy.diff(fy, z)
#         fzz = sympy.diff(fz, z)
#         self.fct_x = np.vectorize(sympy.lambdify('x,y,z', fx),otypes=[float])
#         self.fct_y = np.vectorize(sympy.lambdify('x,y,z', fy),otypes=[float])
#         self.fct_z = np.vectorize(sympy.lambdify('x,y,z', fz),otypes=[float])
#         self.fct_xx = np.vectorize(sympy.lambdify('x,y,z', fxx),otypes=[float])
#         self.fct_xy = np.vectorize(sympy.lambdify('x,y,z', fxy),otypes=[float])
#         self.fct_xz = np.vectorize(sympy.lambdify('x,y,z', fxz),otypes=[float])
#         self.fct_yy = np.vectorize(sympy.lambdify('x,y,z', fyy),otypes=[float])
#         self.fct_yz = np.vectorize(sympy.lambdify('x,y,z', fyz),otypes=[float])
#         self.fct_zz = np.vectorize(sympy.lambdify('x,y,z', fzz),otypes=[float])
#     def __repr__(self):
#         return str(self.expr)
#     def __call__(self, x, y=0, z=0):
#         return self.fct(x,y, z)
#     def x(self, x, y=0, z=0):
#         return self.fct_x(x,y, z)
#     def y(self, x, y, z=0):
#         return self.fct_y(x,y, z)
#     def z(self, x, y, z):
#         return self.fct_z(x,y, z)
#     def xx(self, x, y=0, z=0):
#         return self.fct_xx(x,y,z )
#     def yy(self, x, y, z=0):
#         return self.fct_yy(x,y,z)
#     def zz(self, x, y, z):
#         return self.fct_zz(x,y,z)
#     def d(self, i, x, y, z):
#         if i==2:    return self.fct_z(x,y,z)
#         elif i==1:  return self.fct_y(x,y,z)
#         return self.fct_x(x,y,z)
#     def dd(self, i, j, x, y, z=0):
#         if i==2:
#             if j==2:    return self.fct_zz(x,y,z)
#             elif j==1:  return self.fct_yz(x,y,z)
#             return self.fct_xz(x,y,z)
#         if i==1:
#             if j==2:    return self.fct_yz(x,y,z)
#             elif j==1:  return self.fct_yy(x,y,z)
#             return self.fct_xy(x,y,z)
#         if j==2:    return self.fct_xz(x,y,z)
#         elif j==1:  return self.fct_xy(x,y,z)
#         return self.fct_xx(x,y,z)

#=================================================================#
def analyticalSolution(function, dim, ncomp=1, random=True):
    """
    defines some analytical functions to be used in validation

    returns analytical function (if ncomp==1) or list of analytical functions (if ncomp>1)

    parameters:
        function: name of function
        ncomp: size of list
        random: use random coefficients
    """
    solexact = []
    from itertools import permutations
    def _p(i, n):
        if random:
            p = (4 * np.random.rand(n) - 2) / 3
        else:
            p = [1.1 * (4 - d) for d in range(n)]
        perm = list(permutations(p))
        return perm[i % ncomp]
    vars = ['x', 'y', 'z']
    for i in range(ncomp):
        fct = '{:3.1f}'.format(_p(i,1)[0])
        if function == 'Linear' or function == 'Quadratic':
            p = _p(i, dim)
            for d in range(dim): fct += "+{:3.1f}*{:1s}".format(p[d], vars[d])
            if function == 'Quadratic':
                p = _p(i, dim)
                for d in range(dim): fct += "+{:3.1f}*{:1s}**2".format(p[d], vars[d])
        elif function == 'Sinus':
            p = _p(i, dim)
            for d in range(dim): fct += "+{:3.1f}*sin({:1s})".format(p[d], vars[d])
        else:
            fct = function
        solexact.append(AnalyticalSolution(expr=fct))
    if ncomp==1: return solexact[0]
    return solexact


# ------------------------------------------------------------------- #
if __name__ == '__main__':
    def test1D():
        u = AnalyticalSolution(dim=1, expr='x*x')
        print("u(2)", u(2))
        x = np.meshgrid(np.linspace(0, 2, 3))
        print("x", x, "\nu=", u.expr, "\nu(x)", u(x), "\nu.x(x)", u.x(x), "\nu.xx(x)", u.xx(x))
    def test2D():
        u = AnalyticalSolution(dim=2, expr='x*x*y + y*y')
        print("u(2,1)", u(2,1))
        print("u(2,1)", u(*(2,1)))
        x = np.meshgrid(np.linspace(0, 2, 3),np.linspace(0, 1, 2))
        print("x", x, "\nu=", u.expr, "\nu(x)", u(*x), "\nu.x(x)", u.x(*x), "\nu.xx(x)", u.xx(*x))
    # test2D()
    test1D()



