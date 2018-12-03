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
    # problem = 'Analytic_Sinus'
    geomname="unitsquare"
    bdrycond={}
    bdrycond[11] = "Neumann"
    bdrycond[22] = "Neumann"
    bdrycond[33] = "Dirichlet"
    bdrycond[44] = "Dirichlet"
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
    p2 =  geometry.add_point([1.0, 1.0, 0.0], h)
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
    fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    bdrycond={}
    bdrycond[11] = "Neumann"
    bdrycond[22] = "Neumann"
    bdrycond[33] = "Dirichlet"
    bdrycond[44] = "Dirichlet"
    def dirichlet(x, y): return 120.
    def neumann(x, y): return 0.
    def rhs(x, y): return 1.
    heat = fempy.applications.heat.Heat(dirichlet=dirichlet, neumann=neumann, rhs=rhs, bdrycond=bdrycond)
    heat.setMesh(mesh)
    point_data, cell_data, info = heat.solve()
    fempy.meshes.plotmesh.meshWithData(mesh, point_data, cell_data)
    plt.show()


#================================================================#
test = "coefs_stat"
# test = "analytic"
cmd = 'test_'+test+'()'
eval(cmd)