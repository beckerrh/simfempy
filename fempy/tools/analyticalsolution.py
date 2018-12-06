# -*- coding: utf-8  -*-
"""

"""

import numpy as np
import sympy

class AnalyticalSolution():
    """
    computes numpy vectorized functions for the function and its dericatives up to two
    for a given expression, derivatives computed with sympy
    """
    def __init__(self, expr):
        (x, y, z) = sympy.symbols('x,y,z')
        self.expr = expr
        self.fct = np.vectorize(sympy.lambdify('x,y,z',expr))
        fx = sympy.diff(expr, x)
        fy = sympy.diff(expr, y)
        fz = sympy.diff(expr, z)
        fxx = sympy.diff(fx, x)
        fxy = sympy.diff(fx, y)
        fxz = sympy.diff(fx, z)
        fyy = sympy.diff(fy, y)
        fyz = sympy.diff(fy, z)
        fzz = sympy.diff(fz, z)
        self.fct_x = np.vectorize(sympy.lambdify('x,y,z', fx),otypes=[float])
        self.fct_y = np.vectorize(sympy.lambdify('x,y,z', fy),otypes=[float])
        self.fct_z = np.vectorize(sympy.lambdify('x,y,z', fz),otypes=[float])
        self.fct_xx = np.vectorize(sympy.lambdify('x,y,z', fxx),otypes=[float])
        self.fct_xy = np.vectorize(sympy.lambdify('x,y,z', fxy),otypes=[float])
        self.fct_xz = np.vectorize(sympy.lambdify('x,y,z', fxz),otypes=[float])
        self.fct_yy = np.vectorize(sympy.lambdify('x,y,z', fyy),otypes=[float])
        self.fct_yz = np.vectorize(sympy.lambdify('x,y,z', fyz),otypes=[float])
        self.fct_zz = np.vectorize(sympy.lambdify('x,y,z', fzz),otypes=[float])
    def __str__(self):
        return str(self.expr)
    def __call__(self, x, y, z=0.0):
        return self.fct(x,y, z)
    def x(self, x, y, z=0.0):
        return self.fct_x(x,y, z)
    def y(self, x, y, z=0.0):
        return self.fct_y(x,y, z)
    def z(self, x, y, z=0.0):
        return self.fct_z(x,y, z)
    def xx(self, x, y, z=0.0):
        return self.fct_xx(x,y,z )
    def yy(self, x, y, z=0.0):
        return self.fct_yy(x,y,z)
    def zz(self, x, y, z=0.0):
        return self.fct_zz(x,y,z)

# ------------------------------------------------------------------- #
if __name__ == '__main__':
    uexact = AnalyticalSolution('x*x+2*y*y')
    for (x,y) in [(0,0), (1,1), (1,0), (0,1)]:
        print('Quadratic function', uexact, ' in x,y=:',x,y, ' equals: ', uexact(x,y))
        print('grad=', uexact.x(x,y), uexact.y(x,y))
        print('laplace=', uexact.xx(x,y),  uexact.yy(x,y))

    
    
