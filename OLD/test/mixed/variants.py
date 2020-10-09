# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

print("simfempypath", simfempypath)

from OLD.simfempy.applications.laplacemixed import LaplaceMixed

#----------------------------------------------------------------#
def getGeometryAndData(geomname = "unitcube", h=None):
    data = OLD.simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    postproc = data.postproc
    if geomname == "unitsquare" or geomname == "equilateral":
        if not h: h = [1.3, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02]
        bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
        if geomname == "unitsquare" :
            geometry = OLD.simfempy.meshes.geomdefs.unitsquare.Unitsquare()
        else:
            geometry = OLD.simfempy.meshes.geomdefs.equilateral.Equilateral()
    elif geomname == "unitcube":
        if not h: h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06]
        bdrycond.set("Dirichlet", [100, 101, 102, 103, 104, 105])
        geometry = OLD.simfempy.meshes.geomdefs.unitcube.Unitcube()
    data.params.scal_glob['kheat'] = 0.123
    return geometry, data, h

# ------------------------------------- #
def test_analytic(exactsolution="Quadratic", geomname="unitsquare", verbose=2, h=None):
    import simfempy.tools.comparemethods
    geometry, data, h = getGeometryAndData(geomname, h)
    laplace = LaplaceMixed(geometry=geometry, problemdata=data, showmesh=False)
    problemdata = laplace.generatePoblemDataForAnalyticalSolution(exactsolution=exactsolution, problemdata=data, random=False)
    # print("exact", problemdata.solexact)
    methods = {}
    linearsolver = 'gmres'
    # methods['RT'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver)
    # methods['RTM'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="L2")
    # methods['RTxTilde'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="RTxTilde")
    # methods['RT_Hat'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="RT_Hat")
    # methods['Hat_RT'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="Hat_RT")
    # methods['RT_Tilde'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="RT_Tilde")
    # methods['Tilde_RT'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="Tilde_RT")
    methods['RT_Bar'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="RT_Bar")
    methods['Bar_RT'] = LaplaceMixed(problemdata=problemdata, linearsolver=linearsolver, massproj="Bar_RT")
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=verbose, niter=3)
    result = comp.compare(geometry=geometry)

# ------------------------------------- #
if __name__ == '__main__':
    # test_analytic(exactsolution="Constant")
    # test_analytic(exactsolution="Linear", geomname="equilateral", verbose=2)
    # test_analytic(exactsolution="Linear", geomname="equilateral", verbose=4)
    test_analytic(exactsolution="Linear", geomname="unitsquare")
    # test_analytic(exactsolution="Linear", geomname="unitcube")
    # test_analytic()
    # test_analytic(geomname="unitcube")
