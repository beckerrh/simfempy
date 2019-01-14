assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications


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
    data = geometryResistance(h=0.005)
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

    fems = ['p1', 'cr1']
    fems = ['p1']
    for fem in fems:
        heat = fempy.applications.heat.Heat(bdrycond=bdrycond, kheat=kheat, postproc=postproc, fem=fem, plotk=True)
        heat.setMesh(mesh)
        point_data, cell_data, info = heat.solve()
        print("time: {}".format(info['timer']))
        print("postproc: {}".format(info['postproc']))
        fempy.meshes.plotmesh.meshWithData(mesh, point_data, cell_data)
        plt.show()



#================================================================#

test_coefs_stat()
