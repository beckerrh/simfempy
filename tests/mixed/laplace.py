# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)
from simfempy.meshes import geomdefs
from simfempy.applications.laplacemixed import LaplaceMixed

# ------------------------------------- #
def test_analytic(exactsolution="Quadratic", geomname="unitsquare", verbose=2):
    import simfempy.tools.comparemethods
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    if geomname == "unitsquare":
        h = [1.0, 0.5, 0.25, 0.125, 0.062, 0.03, 0.015]
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrydn'] = "bdrydn:1000,1001"
        geometry = geomdefs.unitsquare.Unitsquare()
    elif geomname == "unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06]
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrydn'] = "bdrydn:100,105"
        geometry = geomdefs.unitcube.Unitcube()
    laplace = LaplaceMixed(geometry=geometry, showmesh=False)
    problemdata = laplace.generatePoblemData(exactsolution=exactsolution, bdrycond=bdrycond, postproc=postproc)
    methods = {}
    methods['poisson'] = LaplaceMixed(problemdata=problemdata)
    if exactsolution == "Linear": h = h[:-3]
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=verbose)
    result = comp.compare(geometry=geometry, h=h)
    return result[3]['error']['pcL2']

# ------------------------------------- #
if __name__ == '__main__':
    test_analytic()