# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse
import xifunction

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from tools.comparerrors import CompareErrors

# ------------------------------------- #

if __name__ == '__main__':
    problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    # problem = 'Analytic_Exponential'
    # problem = 'Analytic_Linear'
    problem = 'RotStat'
    # problem = 'Ramp'
    alpha = 1.0
    alpha = 0.0
    # beta = lambda x, y: (-y, x)
    beta = lambda x, y: (-np.cos(np.pi * (x + y)), np.cos(np.pi * (x + y)))
    beta = lambda x, y: (-np.sin(np.pi * x) * np.cos(np.pi * y), np.sin(np.pi * y) * np.cos(np.pi * x))
    beta = lambda x, y: (-np.cos(np.pi * x) * np.sin(np.pi * y), np.cos(np.pi * y) * np.sin(np.pi * x))
    beta = None

    methods = {}
    from transportcg1supg import TransportCg1Supg
    from transportcg1supgnew import TransportCg1SupgNew
    from transportcg1patches import TransportCg1Patches
    from transportcg1edges import TransportCg1Edges
    from transportcg1supgdynamic import TransportCg1SupgDynamic

    test = 'centered'
    # test = 'firstorder'
    # test = 'secondorderlin'
    test = 'secondorder'
    test = 'supg'

    if test == 'centered':
        methods['centerdSupg'] = TransportCg1Supg(problem=problem, alpha=alpha, beta=beta, delta = 0.0)
        methods['centerdP'] = TransportCg1Patches(upwind='centered', problem=problem, alpha=alpha, beta=beta)
        methods['centerdE'] = TransportCg1Edges(upwind='centered', problem=problem, alpha=alpha, beta=beta)
    elif test == 'firstorder':
        methods['Plin'] = TransportCg1Patches(upwind='linear', problem=problem, alpha=alpha, beta=beta)
        methods['Elin'] = TransportCg1Edges(upwind='linear', problem=problem, alpha=alpha, beta=beta)
        from transportdg0 import TransportDg0
        methods['Dg0'] = TransportDg0(problem=problem, alpha=alpha, beta=beta)
    elif test == 'secondorderlin':
        methods['Supg'] = TransportCg1Supg(problem=problem, alpha=alpha, beta=beta)
        methods['SupgN'] = TransportCg1SupgNew(problem=problem, alpha=alpha, beta=beta)
        #methods['Plin'] = TransportCg1Patches(upwind='nonlinear', xi='xilin', problem=problem, alpha=alpha, beta=beta)
        #methods['PlinS'] = TransportCg1Patches(upwind='nonlinear', sym=True, xi='xilin', problem=problem, alpha=alpha, beta=beta)
        #methods['Elin'] = TransportCg1Edges(upwind='nonlinear', xi='xilin', problem=problem, alpha=alpha, beta=beta)
    elif test == 'secondorder':
        # methods['Supg'] = TransportCg1SupgNew(problem=problem, alpha=alpha, beta=beta)
        # methods['Supgnl'] = TransportCg1SupgNew(upwind="nonlinear", xi='xi', problem=problem, alpha=alpha, beta=beta)
        methods['pnonlin'] = TransportCg1Patches(upwind='newnonlinear', problem=problem, alpha=alpha, beta=beta)
        methods['supg'] = TransportCg1SupgDynamic(problem=problem, upwind = "supg")
        methods['supgsl'] = TransportCg1SupgDynamic(problem=problem, upwind = "supgsl")
    elif test == 'supg':
        # methods['SupgOld'] = TransportCg1Supg(problem=problem, alpha=alpha, beta=beta, delta = 1.0)
        # methods['SupgNew'] = TransportCg1SupgNew(problem=problem, alpha=alpha, beta=beta, upwind = "patch")
        methods['supg'] = TransportCg1SupgDynamic(problem=problem, upwind = "supg")
        methods['xi'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi')
        # methods['xibis'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xibis')
        # methods['xiter'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xiter')
        # methods['xiquater'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xiquater')
        methods['xi2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi2')
        methods['xi3'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi3')
        # methods['xinew'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xinew')
        # methods['xinew2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xinew2')
        # methods['xisignmin'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xisignmin')
        # methods['xispline'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xispline')

        # methods['SupgSl'] = TransportCg1SupgDynamic(problem=problem, upwind = "nl")
        methods['xisc'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi', shockcapturing='phiabs')
        methods['xisc3'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi3', shockcapturing='phiabs')
        # methods['xisignmin'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xisignmin')
        # methods['xisc'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi', shockcapturing='phiabs')
        # methods['xin2sc'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xinew2', shockcapturing='phiabs')
        # methods['xispline'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xispline')
        # methods['Supgslsym'] = TransportCg1SupgDynamic(problem=problem, upwind = "slsym", xi='xi')
        # methods['xisc'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi', shockcapturing='phiabs')
        # methods['xisc2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi2', shockcapturing='phiabs')
        # methods['xisc2'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi2', shockcapturing='phicon')
        # methods['xiscon'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xi2', shockcapturing='phicon')
        # methods['sc'] = TransportCg1SupgDynamic(problem=problem, upwind = "sl", xi='xiconst', shockcapturing=True)

    compareerrors = CompareErrors(methods, latex=True, vtk=True)
    niter = 4
    h = [0.4*np.power(0.4,i)  for i in range(niter)]
    compareerrors.compare(h=h)
