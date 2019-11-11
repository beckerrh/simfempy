assert __name__ == '__main__'
from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.applications
import pygmsh
import matplotlib.pyplot as plt

#----------------------------------------------------------------#
def createMesh():
    from simfempy.meshes import geomdefs
    holes = []
    holes.append([[-0.5, -0.25], [-0.5, 0.25], [0.5, 0.25], [0.5, -0.25]])
    holes.append([[-0.5, 0.75], [-0.5, 1.25], [0.5, 1.25], [0.5, 0.75]])
    holes.append([[-0.5, -0.75], [-0.5, -1.25], [0.5, -1.25], [0.5, -0.75]])
    geometry = geomdefs.unitsquareholes.Unitsquareholes(rect=(-1.1,1.1,-2,2), holes=holes, h=0.2)
    mesh = pygmsh.generate_mesh(geometry)
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

#----------------------------------------------------------------#
def createData():
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    bdrycond.type[1000] = "Neumann"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1002] = "Neumann"
    bdrycond.type[1003] = "Dirichlet"
    postproc['bdrymean_low'] = "bdrymean:1000"
    postproc['bdrymean_up'] = "bdrymean:1002"
    postproc['bdrydn_left'] = "bdrydn:1003"
    postproc['bdrydn_right'] = "bdrydn:1001"
    bdrycond.fct[1000] = lambda x,y,z, nx, ny, nz: 0
    bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 100
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    bdrycond.check(mesh.bdrylabels.keys())
    # print("bdrycond", bdrycond)
    def kheat(label):
        print(f"label={label}")
        if label==100: return 0.0001
        return 1000.0
    data = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
    data.kheat = kheat
    return data

#----------------------------------------------------------------#
def test(mesh, problemdata):
    fem = 'p1' # or fem = 'cr1
    heat = simfempy.applications.heat.Heat(problemdata=problemdata, fem=fem, plotk=True)
    heat.setMesh(mesh)
    point_data, cell_data, info = heat.solve()
    print("fem={} time: {}".format(fem, info['timer']))
    print("postproc: {}".format(info['postproc']))
    simfempy.meshes.plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data, title=fem)
    plt.show()

#================================================================#

mesh = createMesh()
simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
problemdata = createData()
test(mesh, problemdata)
