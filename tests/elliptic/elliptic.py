from os import sys, path

fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

from simfempy.applications.elliptic import Elliptic
from simfempy.meshes import geomdefs

# ----------------------------------------------------------------#
def test_analytic(problem="Analytic_Quadratic", geomname="unitsquare", verbose=2):
    import simfempy.tools.comparerrors
    bdrycond0 = simfempy.applications.problemdata.BoundaryConditions()
    bdrycond1 = simfempy.applications.problemdata.BoundaryConditions()
    postproc0 = {}
    postproc1 = {}
    if geomname == "unitsquare":
        h = [0.5, 0.25, 0.125, 0.06, 0.03]
        if problem=="Analytic_Linear": h = h[:-2]
        problem += '_2d'
        bdrycond0.type[1000] = "Neumann"
        bdrycond0.type[1001] = "Dirichlet"
        bdrycond0.type[1002] = "Neumann"
        bdrycond0.type[1003] = "Dirichlet"
        postproc0['bdrymean'] = "bdrymean:1000,1002"
        postproc0['bdrydn'] = "bdrydn:1001,1003"
        bdrycond1.type[1000] = "Dirichlet"
        bdrycond1.type[1001] = "Neumann"
        bdrycond1.type[1002] = "Dirichlet"
        bdrycond1.type[1003] = "Neumann"
        postproc1['bdrymean'] = "bdrymean:1001,1003"
        postproc1['bdrydn'] = "bdrydn:1000,1002"
    elif geomname == "unitcube":
        h = [0.5, 0.25, 0.125, 0.06]
        if problem=="Analytic_Linear": h = h[:-2]
        problem += '_3d'
        bdrycond0.type[100] = "Neumann"
        bdrycond0.type[101] = "Dirichlet"
        bdrycond0.type[102] = "Dirichlet"
        bdrycond0.type[103] = "Dirichlet"
        bdrycond0.type[104] = "Dirichlet"
        bdrycond0.type[105] = "Neumann"
        postproc0['bdrymean'] = "bdrymean:100,105"
        postproc0['bdrydn'] = "bdrydn:101,102,103,104"
        bdrycond1.type[100] = "Dirichlet"
        bdrycond1.type[101] = "Neumann"
        bdrycond1.type[102] = "Dirichlet"
        bdrycond1.type[103] = "Neumann"
        bdrycond1.type[104] = "Dirichlet"
        bdrycond1.type[105] = "Dirichlet"
        postproc1['bdrymean'] = "bdrymean:101,103"
        postproc1['bdrydn'] = "bdrydn:100,102,104,105"

    bdrycond = [bdrycond0, bdrycond1]
    postproc = [postproc0, postproc1]
    compares = {}
    app = Elliptic(problem=problem, bdrycond=bdrycond, ncomp=2)
    for fem in ['p1', 'cr1']:
        for bdry in ['trad', 'new']:
            compares[fem + bdry] = Elliptic(solexact=app.solexact, bdrycond=bdrycond, postproc=postproc, fem=fem,\
                                            ncomp=2, method=bdry, problemname=app.problemname)
    comp = simfempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    if problem.split('_')[1] == "Linear":
        h = [2, 1, 0.5, 0.25]
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['L2']


# ================================================================#
if __name__ == '__main__':
    test_analytic(problem="Analytic_Linear", geomname="unitcube")
