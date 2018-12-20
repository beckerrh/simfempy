assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications

#----------------------------------------------------------------#
def test_flux():
    import fempy.tools.comparerrors
    geomname = "unitsquare"
    geomname = "unitcube"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Dirichlet"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.fct[11] = lambda x,y,z: 0
    bdrycond.fct[44] = bdrycond.fct[33] = bdrycond.fct[22] = bdrycond.fct[11]
    postproc = {}
    postproc['flux'] = "flux:11,22,33,44"
    if geomname == "unitcube":
        bdrycond.type[55] = "Dirichlet"
        bdrycond.type[66] = "Dirichlet"
        bdrycond.fct[66] = bdrycond.fct[55] = bdrycond.fct[44]
        postproc['flux'] += ",55,66"
    methods = {}
    for method in ['cr1-trad', 'cr1-new']:
        fem, meth  = method.split('-')
        methods[method] = fempy.applications.heat.Heat(rhs=lambda x,y,z:1, bdrycond=bdrycond, kheat=lambda id:1, postproc=postproc, fem=fem, method=meth)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    result = comp.compare(geomname=geomname, h=[2, 1, 0.5, 0.25, 0.125, 0.06, 0.03])

#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[33] = "Neumann"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,33"
    postproc['flux'] = "flux:22,44"
    methods = {}
    for fem in ['p1', 'cr1']:
        methods[fem] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [0.5, 0.25, 0.125, 0.06, 0.03]
    # h = [2.0, 1.0]
    result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def test_analytic3d():
    import matplotlib.pyplot as plt
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear3d'
    problem = 'Analytic_Quadratic3d'
    problem = 'Analytic_Sinus3d'
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,22"
    postproc['flux'] = "flux:33,44,55,66"
    methods = {}
    for fem in ['p1', 'cr1']:
        methods[fem] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [np.power(i*20,-2/3) for i in range(1,6)]
    h = [1.0, 0.5, 0.25, 0.13, 0.08]
    print("h", h)
    result = comp.compare(geomname="unitcube", h=h)


# ----------------------------------------------------------------#
def test_solvers():
    import time
    problem = 'Analytic_Sinus3d'
    bdrycond = fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,22"
    postproc['flux'] = "flux:33,44,55,66"
    heat  = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc, fem=fem)
    mesh = fempy.meshes.simplexmesh.SimplexMesh(geomname='unitcube', hmean=0.05)
    heat.setMesh(mesh)
    print("heat.linearsolvers=", heat.linearsolvers)
    b = heat.computeRhs()
    A = heat.matrix()
    A, b = heat.boundary(A, b)
    for solver in heat.linearsolvers:
        t0 = time.time()
        u = heat.linearSolver(A, b, solver=solver)
        t1 = time.time()
        print("{:4d} {:12s} {:10.2e}".format(mesh.ncells, solver, t1-t0))


#----------------------------------------------------------------#
def geometryResistance(h=0.04):
    import pygmsh
    geometry = pygmsh.built_in.Geometry()
    a, b = 1.0, 2.0
    d, e = 0.5, 0.25
    h1 = geometry.add_rectangle(-d, d, -e, e, 0, lcar=h)
    geometry.add_physical_surface(h1.surface, label=222)
    h2 = geometry.add_rectangle(-d, d, -e+1, e+1, 0, lcar=h)
    geometry.add_physical_surface(h2.surface, label=333)
    p1 = geometry.add_rectangle(-a, a, -b, b, 0, lcar=h, holes=[h1,h2])
    geometry.add_physical_surface(p1.surface, label=111)
    for i in range(4): geometry.add_physical_line(p1.line_loop.lines[i], label=11*(1+i))
    code = geometry.get_code()
    print("code:\n", code)
    return pygmsh.generate_mesh(geometry)

#----------------------------------------------------------------#
def test_coefs_stat():
    import matplotlib.pyplot as plt
    data = geometryResistance(h=0.07)
    # points, cells, celldata = data[0], data[1], data[2]
    # plt.triplot(points[:,0], points[:,1], cells['triangle'])
    # plt.show()

    mesh = fempy.meshes.simplexmesh.SimplexMesh(data=data)
    fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions(mesh.bdrylabels.keys())
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[33] = "Neumann"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.fct[11] = lambda x,y,z, nx, ny, nz, k: 0
    bdrycond.fct[33] = lambda x,y,z, nx, ny, nz, k: 100
    bdrycond.fct[22] = lambda x,y,z: 120
    bdrycond.fct[44] = bdrycond.fct[22]
    # print("bdrycond", bdrycond)
    rhs = lambda x, y, z: 0.
    def kheat(label):
        if label==111: return 0.1
        return 10000.0

    postproc = {}
    postproc['mean11'] = "mean:11"
    postproc['mean33'] = "mean:33"
    postproc['flux22'] = "flux:22"
    postproc['flux44'] = "flux:44"

    for fem in ['p1', 'cr1']:
        heat = fempy.applications.heat.Heat(bdrycond=bdrycond, kheat=kheat, postproc=postproc, fem=fem)
        heat.setMesh(mesh)
        point_data, cell_data, info = heat.solvestatic()
        print("time: {}".format(info['timer']))
        print("postproc: {}".format(info['postproc']))
        fempy.meshes.plotmesh.meshWithData(mesh, point_data, cell_data)
        plt.show()



#================================================================#

#test_analytic()
#test_analytic3d()
#test_solvers()
test_flux()
#test_coefs_stat()
