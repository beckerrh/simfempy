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
    # bdrycond.type[11] = "Neumann"
    # bdrycond.type[22] = "Neumann"
    bdrycond.type[11] = "Dirichlet"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    bdrycond.fct[11] = lambda x,y,z: 0
    bdrycond.fct[66] = bdrycond.fct[55] = bdrycond.fct[44] = bdrycond.fct[33] = bdrycond.fct[22] = bdrycond.fct[11]
    postproc = {}
    # postproc['mean'] = "11,22"
    postproc['flux'] = "flux:11,22,33,44,55,66"
    methods = {}
    methods['p1'] = fempy.applications.heat.Heat(rhs=lambda x,y,z:1, bdrycond=bdrycond, kheat=lambda id:1, postproc=postproc)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    result = comp.compare(geomname=geomname, h=[2.0, 1.0, 0.5, 0.25, 0.125, 0.06])

#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
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
    methods['p1'] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06, 0.03]
    result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def test_analytic3d():
    import matplotlib.pyplot as plt
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear3d'
    problem = 'Analytic_Quadratic3d'
    # problem = 'Analytic_Sinus3d'

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
    methods['p1'] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc)

    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    h = [np.power(i*20,-2/3) for i in range(1,6)]
    h = [1.0, 0.5, 0.25, 0.13, 0.08, 0.05]
    print("h", h)
    result = comp.compare(geomname="unitcube", h=h)

#----------------------------------------------------------------#
def geometryResistance():
    import pygmsh
    geometry = pygmsh.built_in.Geometry()
    h = 0.05
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
    data = geometryResistance()
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
        return 100.0

    postproc = {}
    postproc['mean11'] = "mean:11"
    postproc['mean33'] = "mean:33"
    postproc['flux22'] = "flux:22"
    postproc['flux44'] = "flux:44"
    heat = fempy.applications.heat.Heat(rhs=rhs, bdrycond=bdrycond, kheat=kheat, postproc=postproc)
    heat.setMesh(mesh)
    point_data, cell_data, info = heat.solvestatic()
    print("time: {}".format(info['timer']))
    print("postproc: {}".format(info['postproc']))
    fempy.meshes.plotmesh.meshWithData(mesh, point_data, cell_data)
    plt.show()



#================================================================#

#test_analytic()
test_analytic3d()
#test_flux()
#test_coefs_stat()
