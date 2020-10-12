import simfempy
import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.heat import Heat

#----------------------------------------------------------------#
def getGeometryAndData(geomname = "unitcube"):
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    postproc = data.postproc
    if geomname == "unitline":
        createMesh = testmeshes.unitline
        bdrycond.set("Dirichlet", [10000,10001])
    elif geomname == "unitsquare":
        bdrycond.set("Neumann", [1000, 1002])
        bdrycond.set("Dirichlet", [1001])
        bdrycond.set("Robin", [1003])
        bdrycond.param[1003] = 11
        postproc.type['bdrymean'] = "bdrymean"
        postproc.color['bdrymean'] = [1000, 1002]
        postproc.type['fluxn'] = "bdrydn"
        postproc.color['fluxn'] = [1001, 1003]
        createMesh = testmeshes.unitsquare
    elif geomname == "unitcube":
        bdrycond.set("Neumann", [100, 105])
        bdrycond.set("Dirichlet", [101, 102])
        bdrycond.set("Robin", [103, 104])
        bdrycond.param[103] = 1
        bdrycond.param[104] = 10
        postproc.type['bdrymean'] = "bdrymean"
        postproc.color['bdrymean'] = [100, 105]
        postproc.type['fluxn'] = "bdrydn"
        postproc.color['fluxn'] = [101,102,103,104]
        createMesh = testmeshes.unitcube
    data.params.scal_glob['kheat'] = 0.123
    return createMesh, data

#----------------------------------------------------------------#
def test_analytic(exactsolution="Linear", geomname = "unitsquare", verbose=1, fems=['p1'], methods=['trad']):
    import simfempy.tools.comparemethods
    createMesh, data = getGeometryAndData(geomname)
    h = [0.5, 0.25, 0.125, 0.06, 0.03, 0.02]
    if geomname == "unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.125]
    if exactsolution == "Linear":  h = h[:-3]
    heat = Heat(mesh=createMesh(h[0]), problemdata=data)
    colors = [c for c in data.bdrycond.colors()]
    data.bdrycond.clear()
    data.bdrycond.set("Neumann", [colors[0]])
    # data.bdrycond.set("Robin", [colors[1]])
    # data.bdrycond.param[colors[1]] = 1000000
    # data.bdrycond.set("Dirichlet", colors[2:])
    data.bdrycond.set("Dirichlet", colors[1:])
    problemdata = heat.generatePoblemDataForAnalyticalSolution(exactsolution=exactsolution, problemdata=data, random=False)
    sims = {}
    for fem in fems:
        for method in methods:
            sims[fem+method] = Heat(problemdata=problemdata, fem=fem, method=method)
    comp = simfempy.tools.comparemethods.CompareMethods(sims, verbose=verbose)
    result = comp.compare(createMesh=createMesh, h=h)
    return result[3]['error']

#----------------------------------------------------------------#
def test_flux(geomname = "unitcube"):
    import simfempy.tools.comparemethods
    geometry, data = getGeometryAndData(geomname)
    colors = [c for c in data.bdrycond.colors()]
    data.bdrycond.clear()
    data.bdrycond.set("Dirichlet", colors, len(colors)*[lambda x,y,z: 0])
    data.postproc.clear()
    data.postproc.type['flux'] = "bdrydn"
    data.postproc.color['flux'] = colors
    data.rhs = lambda x, y, z:1
    data.params.scal_glob['kheat'] = 1
    methods = {}
    for method in ['p1-trad', 'p1-new', 'cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = Heat(problemdata=data, fem=fem, method=meth, linearsolver='umf')
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=2)
    h = [2, 1, 0.5, 0.25, 0.125]
    result = comp.compare(geometry=geometry, h=h)

# ----------------------------------------------------------------#
def test_solvers(geomname='unitcube', fem = 'p1', method='trad'):
    exactsolution = 'Sinus'
    import simfempy
    geometry, data = getGeometryAndData(geomname)
    if geomname == "unitsquare":
        hmean = 0.06
    elif geomname == "unitcube":
        hmean = 0.08
    heat = Heat(geometry=geometry, problemdata=data)
    problemdata = heat.generatePoblemDataForAnalyticalSolution(exactsolution=exactsolution, problemdata=data, random=False)
    heat  = Heat(problemdata=problemdata, fem=fem, method=method)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=hmean)
    heat.setMesh(mesh)
    print(f"N = {mesh.ncells} heat.linearsolvers={heat.linearsolvers}")
    A = heat.matrix()
    b,u = heat.computeRhs()
    import simfempy.tools.timer
    timer = simfempy.tools.timer.Timer(name=fem + '_' + geomname + '_' + str(mesh.ncells))
    for solver in heat.linearsolvers:
        u, niter = heat.linearSolver(A, b, solver=solver, verbose=20)
        timer.add(solver)
    print(f"{timer}")

#----------------------------------------------------------------#
def test_dirichlet(exactsolution="Linear", geomname = "unitsquare", verbose=3):
    import simfempy.tools.comparemethods
    geometry, data = getGeometryAndData(geomname)
    if geomname == "unitsquare":
        h = 0.06
        h = 1
    elif geomname == "unitcube":
        h = 0.125
    heat = Heat(geometry=geometry, problemdata=data)
    colors = [c for c in data.bdrycond.colors()]
    data.bdrycond.clear()
    data.bdrycond.set("Neumann", [colors[0]])
    data.bdrycond.set("Robin", [colors[1]])
    data.bdrycond.param[colors[1]] = 1.2
    data.bdrycond.set("Dirichlet", colors[2:])
    problemdata = heat.generatePoblemDataForAnalyticalSolution(exactsolution=exactsolution, problemdata=data, random=False)
    method = 'p1-new'
    fem, meth  = method.split('-')
    methods = {method: Heat(problemdata=problemdata, fem=fem, method=meth)}
    comp = simfempy.tools.comparemethods.CompareMethods(methods, h=h, paramname='dirichlet_al', verbose=verbose)
    params = [1, 2, 4, 10, 100,1000]
    result = comp.compare(geometry=geometry, params=params)

#================================================================#
if __name__ == '__main__':
    # test_analytic(exactsolution = 'Constant', geomname = "unitsquare", verbose=4)
    # test_analytic(exactsolution = 'Linear', geomname = "unitline")
    test_analytic(exactsolution = 'Linear', geomname = "unitsquare")
    # test_analytic(exactsolution = 'Quadratic', geomname = "unitsquare", fems= ['p1','cr1'], methods=['trad', 'new'])
    # test_analytic(exactsolution = 'Sinus', geomname = "unitsquare")

    # test_analytic(exactsolution = 'Linear', geomname = "unitcube")
    # test_analytic(exactsolution = 'Quadratic', geomname = "unitcube")

    # test_dirichlet(exactsolution = 'Linear', geomname = "unitsquare")
    # test_analytic(exactsolution = 'Quadratic', geomname = "unitcube")

    # test_solvers(geomname='unitsquare')
    # test_solvers()

    # test_flux()
