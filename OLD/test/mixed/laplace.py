# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

from OLD.simfempy.applications.laplacemixed import LaplaceMixed


#----------------------------------------------------------------#
def getGeometryAndData(geomname = "unitcube"):
    data = OLD.simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    postproc = data.postproc
    if geomname == "unitsquare" or geomname == "equilateral":
        bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
        # bdrycond.set("Dirichlet", [1000, 1001])
        # bdrycond.set("Neumann", [1002])
        # bdrycond.set("Robin", [1003])
        # bdrycond.param[1003] = 11
        # postproc.type['bdrymean'] = "bdrymean"
        # postproc.color['bdrymean'] = [1000, 1002]
        # postproc.type['fluxn'] = "bdrydn"
        # postproc.color['fluxn'] = [1001, 1003]
        if geomname == "unitsquare" :
            geometry = OLD.simfempy.meshes.geomdefs.unitsquare.Unitsquare()
        else:
            # h = None
            # niter = 3
            geometry = OLD.simfempy.meshes.geomdefs.equilateral.Equilateral()
    elif geomname == "unitcube":
        bdrycond.set("Dirichlet", [100, 105])
        bdrycond.set("Neumann", [101, 102])
        bdrycond.set("Robin", [103, 104])
        bdrycond.param[103] = 1
        bdrycond.param[104] = 10
        postproc.type['bdrymean'] = "bdrymean"
        postproc.color['bdrymean'] = [101, 102]
        postproc.type['fluxn'] = "bdrydn"
        postproc.color['fluxn'] = [100,103,104,105]
        geometry = OLD.simfempy.meshes.geomdefs.unitcube.Unitcube()
    data.params.scal_glob['kheat'] = 0.123
    return geometry, data

# ------------------------------------- #
def test_analytic(exactsolution="Quadratic", geomname="unitsquare", verbose=2):
    import simfempy.tools.comparemethods
    geometry, data = getGeometryAndData(geomname)
    laplace = LaplaceMixed(geometry=geometry, problemdata=data)
    problemdata = laplace.generatePoblemDataForAnalyticalSolution(exactsolution=exactsolution, problemdata=data, random=False)
    print("exact", problemdata.solexact)
    methods = {}
    linearsolver = 'gmres'
    h = [1.0, 0.5, 0.25, 0.125]
    methods['RT'] = LaplaceMixed(problemdata=problemdata, fem='rt0', linearsolver=linearsolver)
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=verbose)
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