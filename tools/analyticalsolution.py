# -*- coding: utf-8  -*-
"""

"""

import numpy as np
import sympy

class AnalyticalSolution():
    def __init__(self, expr):
        (x, y) = sympy.symbols('x,y')
        self.expr = expr
        self.fct = np.vectorize(sympy.lambdify('x,y',expr))
        fx = sympy.diff(expr, x)
        fy = sympy.diff(expr, y)
        fxx = sympy.diff(fx, x)
        fxy = sympy.diff(fx, y)
        fyy = sympy.diff(fy, y)
        self.fct_x = np.vectorize(sympy.lambdify('x,y', fx))
        self.fct_y = np.vectorize(sympy.lambdify('x,y', fy))
        self.fct_xx = np.vectorize(sympy.lambdify('x,y', fxx))
        self.fct_xy = np.vectorize(sympy.lambdify('x,y', fxy))
        self.fct_yy = np.vectorize(sympy.lambdify('x,y', fyy))
    def __str__(self):
        return str(self.expr)
    def __call__(self, x, y):
        return self.fct(x,y)
    def x(self, x, y):
        return self.fct_x(x,y)
    def y(self, x, y):
        return self.fct_y(x,y)
    def xx(self, x, y):
        return self.fct_xx(x,y)
    def yy(self, x, y):
        return self.fct_yy(x,y)

# ------------------------------------------------------------------- #
if __name__ == '__main__':
    uexact = AnalyticalSolution('x*x+2*y*y')
    for (x,y) in [(0,0), (1,1), (1,0), (0,1)]:
        print 'Quadratic function', uexact, ' in x,y=:',x,y, ' equals: ', uexact(x,y)
        print 'grad=', uexact.x(x,y), uexact.y(x,y)
        print 'laplace=', uexact.xx(x,y),  uexact.yy(x,y)

    
    
