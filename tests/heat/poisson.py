assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications

#----------------------------------------------------------------#
def test_analytic():
    import fempy.tools.comparerrors
    problem = 'Analytic_Linear'
    # problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'
    geomname = "unitsquare"
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Neumann"
    bdrycond.type[33] = "Dirichlet"
    bdrycond.type[44] = "Dirichlet"
    methods = {}
    methods['p1'] = fempy.applications.heat.Heat(problem=problem, bdrycond=bdrycond)
    comp = fempy.tools.comparerrors.CompareErrors(methods, latex=True, vtk=True, plot=True)
    # comp.compare(h=[1.0, 0.5, 0.25, 0.125])
    comp.compare(geomname=geomname, h=[2, 1.0, 0.5, 0.25, 0.125])

#----------------------------------------------------------------#
def test_coefs_stat():
    import pygmsh
    import fempy.meshes
    import matplotlib.pyplot as plt
    geometry = pygmsh.built_in.Geometry()
    h = 0.05
    p0 =  geometry.add_point([-2.0, -2.0, 0.0], h)
    p1 =  geometry.add_point([1.0, -1.0, 0.0], h)
    p2 =  geometry.add_point([2.0, 2.0, 0.0], h)
    p3 =  geometry.add_point([-1.0, 1.0, 0.0], h)
    l0 =  geometry.add_line(p0, p1)
    l1 =  geometry.add_line(p1, p2)
    l2 =  geometry.add_line(p2, p3)
    l3 =  geometry.add_line(p3, p0)
    ll =  geometry.add_line_loop([l0, l1, l2, l3])
    surf =  geometry.add_plane_surface(ll)
    pl0 =  geometry.add_physical_line(l0, label=11)
    pl1 =  geometry.add_physical_line(l1, label=22)
    pl1 =  geometry.add_physical_line(l2, label=33)
    pl4 =  geometry.add_physical_line(l3, label=44)
    geometry.add_physical_surface(surf, label=99)
    data = pygmsh.generate_mesh(geometry)
    mesh = fempy.meshes.trianglemesh.TriangleMesh(data=data)
    # fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions(mesh.bdrylabelsmsh)
    bdrycond.type[11] = "Neumann"
    bdrycond.type[22] = "Dirichlet"
    bdrycond.type[33] = "Neumann"
    bdrycond.type[44] = "Dirichlet"
    bdrycond.fct[11] = lambda x,y, nx, ny, k: 0
    bdrycond.fct[33] = lambda x,y, nx, ny, k: -10
    bdrycond.fct[22] = lambda x,y: 120
    bdrycond.fct[44] = bdrycond.fct[22]
    print("bdrycond", bdrycond)
    rhs = lambda x, y: 20.
    def kheat(x, y):
        if (-0.25 < x < 0.25) and (0.25 < y < 0.75):
            return 1234.5
        else:
            return 0.123

    heat = fempy.applications.heat.Heat(rhs=rhs, bdrycond=bdrycond, kheat=kheat)
    heat.setMesh(mesh)
    point_data, cell_data, info = heat.solve()
    fempy.meshes.plotmesh.meshWithData(mesh, point_data, cell_data)
    plt.show()


#================================================================#
test = "coefs_stat"
# test = "analytic"
cmd = 'test_'+test+'()'
eval(cmd)