# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
from tools.analyticalsolution import AnalyticalSolution
from fem.femp12d import FemP12D
from transport import Transport
from mesh.trimesh import TriMesh
from tools.comparerrors import CompareErrors
from transportdg0 import TransportDg0
from transportcg1 import TransportCg1

problem = 'Analytic_Quadratic'
problem = 'Analytic_Sinus'
problem = 'Analytic_Exponential'
problem = 'Analytic_Linear'
# problem = 'RotStat'
# problem = 'Ramp'
alpha = 1.0
alpha = 0.0
# beta = lambda x, y: (-y, x)
beta = lambda x, y: (-np.cos(np.pi*(x+y)), np.cos(np.pi*(x+y)))
beta = lambda x, y: (-np.sin(np.pi*x)*np.cos(np.pi*y), np.sin(np.pi*y)*np.cos(np.pi*x))
beta = lambda x, y: (-np.cos(np.pi*x)*np.sin(np.pi*y), np.cos(np.pi*y)*np.sin(np.pi*x))
beta = None

methods = {}
# first-order
# methods['dg0'] = TransportDg0(problem=problem, alpha=alpha, beta=beta)
# methods['cg1uppatch'] = TransportCg1(upwind='mon', problem=problem, alpha=alpha, beta=beta)
# methods['cg1up'] = TransportCg1(upwind='monedge', problem=problem, alpha=alpha, beta=beta)
methods['cg1upmin'] = TransportCg1(upwind='monedge', diff ='min', problem=problem, alpha=alpha, beta=beta)

methods['supg'] = TransportCg1(upwind='supg', problem=problem, alpha=alpha, beta=beta)
methods['nlpatch'] = TransportCg1(upwind='nl', diff ='min', problem=problem, alpha=alpha, beta=beta)
methods['nlmin'] = TransportCg1(upwind='nledge', diff ='min', problem=problem, alpha=alpha, beta=beta)
# methods['nl'] = TransportCg1(upwind='nledge', problem=problem, alpha=alpha, beta=beta)

compareerrors = CompareErrors(methods, latex=True)
compareerrors.compare(orders=[1], h=[1.0, 0.5, 0.25, 0.1], vtk=True)
# compareerrors.compare(h=[1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.013])
