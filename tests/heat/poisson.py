assert __name__ == '__main__'
from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications

#----------------------------------------------------------------#
def test_flux():
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    # bdrycond.type[11] = "Neumann"
    # bdrycond.type[22] = "Neumann"
    bdrycond.type[11] = "Dirichlet"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.fct[11] = lambda x,y,z: 0
    bdrycond.fct[44] = bdrycond.fct[33] = bdrycond.fct[22] = bdrycond.fct[11]
    postproc = {}
    # postproc['mean'] = "11,22"
    postproc['flux'] = "flux:11,22,33,44"
    methods = {}
    methods['p1'] = fempy.applications.heat.Heat(rhs=lambda x,y,z:1, bdrycond=bdrycond, kheat=lambda id:1, postproc=postproc)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    result = comp.compare(geomname=geomname, h=[2.0, 1.0, 0.5, 0.25, 0.125, 0.06, 0.03])

#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    # problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,22"
    postproc['flux'] = "flux:33,44"
    methods = {}
    methods['p1'] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc)
    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=True)
    h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.01]
    result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def test_analytic3d():
    import matplotlib.pyplot as plt
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    # problem = 'Analytic_Sinus'

    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.type[55] = "Dirichlet"
    bdrycond.type[66] = "Dirichlet"
    postproc = {}
    postproc['mean'] = "mean:11,22"
    postproc['flux'] = "flux:33,44"
    methods = {}
    methods['p1'] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond, postproc=postproc)

    mesh = fempy.meshes.simplexmesh.SimplexMesh(geomname="unitcube", hmean=0.1)
    # fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    methods['p1'].setMesh(mesh)
    point_data, cell_data, info = methods['p1'].solvestatic()
    print("info", info)
    # comp = fempy.tools.comparerrors.CompareErrors(methods, plot=True)
    h = [2.0, 1.0, 0.5, 0.25]
    # h = [2.0, 1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.01]
    # h = [2.0, 1.0]
    # result = comp.compare(geomname=geomname, h=h)

#----------------------------------------------------------------#
def linelooprectangle(geometry, x0, y0, a, b, h, colors= 11*np.arange(1,5)):
    p = []
    for i in range(4):
        s1 = 2* ( (i+1)//2 % 2 )-1
        s2 = 2* (i//2)-1
        p.append(geometry.add_point([x0+s1*a, y0+s2*b, 0.0], h))
    l = []
    for i in range(4):
        l.append(geometry.add_line(p[i], p[(i+1)%4]))
        if colors is not None: geometry.add_physical_line(l[i], label=int(colors[i]))
    return geometry.add_line_loop(l)

def geometryResistance():
    import pygmsh
    geometry = pygmsh.built_in.Geometry()
    h = 0.05
    a, b = 1.0, 2.0
    ll = linelooprectangle(geometry, 0, 0, a, b, h)
    d, e = 0.5, 0.25
    mm = linelooprectangle(geometry, 0, 1.0, d, e, h, colors=None)
    nn = linelooprectangle(geometry, 0, 0.0, d, e, h, colors=None)
    surf =  geometry.add_plane_surface(ll, [mm, nn])
    geometry.add_physical_surface(surf, label=111)
    hole1 =  geometry.add_plane_surface(mm)
    geometry.add_physical_surface(hole1, label=222)
    hole2 =  geometry.add_plane_surface(nn)
    geometry.add_physical_surface(hole2, label=333)
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
#test_analytic3d()
#test_flux()
test_coefs_stat()
