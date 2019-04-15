# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)
from simfempy.meshes import geomdefs
from simfempy.applications.laplacemixed import LaplaceMixed

# ------------------------------------- #
def test_analytic(exactsolution="Quadratic", geomname="unitsquare", verbose=2):
    import simfempy.tools.comparemethods
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    niter = None
    if geomname == "unitsquare" or geomname == "equilateral":
        h = [1.3, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02]
        # h = [1.0, 0.5, 0.25]
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        # bdrycond.type[1002] = "Neumann"
        # bdrycond.type[1003] = "Robin"
        # bdrycond.param[1003] = 11
        # postproc['bdrydn'] = "bdrydn:1000,1001"
        # postproc['bdrymean'] = "bdrymean:1002"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        if geomname == "unitsquare" :
            geometry = geomdefs.unitsquare.Unitsquare()
        else:
            # h = None
            # niter = 3
            geometry = geomdefs.equilateral.Equilateral()
    elif geomname == "unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06]
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Neumann"
        bdrycond.type[102] = "Neumann"
        bdrycond.type[103] = "Robin"
        bdrycond.type[104] = "Robin"
        bdrycond.param[103] = 1
        bdrycond.param[104] = 10
        postproc['bdrydn'] = "bdrydn:100,105"
        postproc['bdrymean'] = "bdrymean:101,102"
        geometry = geomdefs.unitcube.Unitcube()
    laplace = LaplaceMixed(geometry=geometry, showmesh=False)
    problemdata = laplace.generatePoblemData(exactsolution=exactsolution, bdrycond=bdrycond, postproc=postproc, random=False)
    print("exact", problemdata.solexact)
    methods = {}
    linearsolver = 'gmres'
    methods['RT'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver)
    methods['RTM'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver, massproj="L2")
    methods['TildexRT'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver, massproj="Tilde-RT")
    methods['RTxTilde'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver, massproj="RT-Tilde")
    # methods['HatxRT'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver, massproj="Hat-RT")
    # methods['TildexTilde'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver, massproj="Tilde-Tilde")
    # methods['BV'] = LaplaceMixed(problemdata=problemdata, fem='bv0', linearsolver=linearsolver)
    # if exactsolution == "Linear" or exactsolution == "Constant": h = h[:-5]
    # h = [2]
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=4, niter=niter)
    result = comp.compare(geometry=geometry, h=h)
    return result[3]['error']['pcL2']

# ------------------------------------- #
if __name__ == '__main__':
    # test_analytic(exactsolution="Constant")
    # test_analytic(exactsolution="Linear", geomname="equilateral")
    # test_analytic(exactsolution="Linear", geomname="unitsquare")
    test_analytic(exactsolution="Quadratic")
    # test_analytic(exactsolution="Linear", geomname="unitcube")
    # test_analytic()
    # test_analytic(geomname="unitcube")