assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications
import pygmsh
import matplotlib.pyplot as plt

#----------------------------------------------------------------#
def test():
    from fempy.meshes.geomdefs import unitsquareholes
    holes = []
    holes.append([[-0.5, -0.25], [-0.5, 0.25], [0.5, 0.25], [0.5, -0.25]])
    holes.append([[-0.5, 0.75], [-0.5, 1.25], [0.5, 1.25], [0.5, 0.75]])
    holes.append([[-0.5, -0.75], [-0.5, -1.25], [0.5, -1.25], [0.5, -0.75]])
    geometry = unitsquareholes.define_geometry(rect=(-1,1,-2,2), holes=holes, h=0.2)
    data = pygmsh.generate_mesh(geometry)

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
        if label==100: return 0.1
        return 10000.0

    postproc = {}
    postproc['mean11'] = "bdrymean:11"
    postproc['mean33'] = "bdrymean:33"
    postproc['flux22'] = "bdrydn:22"
    postproc['flux44'] = "bdrydn:44"

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

test()
