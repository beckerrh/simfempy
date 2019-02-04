assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import simfempy.applications
import pygmsh
import matplotlib.pyplot as plt

#----------------------------------------------------------------#
def test():
    from simfempy.meshes import geomdefs
    holes = []
    holes.append([[-0.5, -0.25], [-0.5, 0.25], [0.5, 0.25], [0.5, -0.25]])
    holes.append([[-0.5, 0.75], [-0.5, 1.25], [0.5, 1.25], [0.5, 0.75]])
    holes.append([[-0.5, -0.75], [-0.5, -1.25], [0.5, -1.25], [0.5, -0.75]])
    # geometry = unitsquareholes.define_geometry(rect=(-1,1,-2,2), holes=holes, h=0.2)
    geometry = geomdefs.unitsquareholes.Unitsquareholes(rect=(-1,1,-2,2), holes=holes, h=0.2)
    data = pygmsh.generate_mesh(geometry)

    mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
    simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions(mesh.bdrylabels.keys())
    postproc = {}
    bdrycond.type[1000] = "Neumann"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1002] = "Neumann"
    bdrycond.type[1003] = "Dirichlet"
    postproc['bdrymean_low'] = "bdrymean:1000"
    postproc['bdrymean_up'] = "bdrymean:1002"
    postproc['bdrydn_left'] = "bdrydn:1003"
    postproc['bdrydn_right'] = "bdrydn:1001"
    bdrycond.fct[1000] = lambda x,y,z, nx, ny, nz, k: 0
    bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz, k: 100
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    # print("bdrycond", bdrycond)
    def kheat(label):
        if label==100: return 0.1
        return 10000.0
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond)

    fems = ['p1', 'cr1']
    fems = ['p1']
    for fem in fems:
        heat = simfempy.applications.heat.Heat(problemdata=problemdata, kheat=kheat, postproc=postproc, fem=fem, plotk=True)
        heat.setMesh(mesh)
        point_data, cell_data, info = heat.solve()
        print("time: {}".format(info['timer']))
        print("postproc: {}".format(info['postproc']))
        simfempy.meshes.plotmesh.meshWithData(mesh, point_data, cell_data)
        plt.show()

#================================================================#

test()
