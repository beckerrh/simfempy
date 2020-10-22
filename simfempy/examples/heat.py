assert __name__ == '__main__'
import os, sys
# simfempypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(simfempypath)

import simfempy
from simfempy.meshes.hole import square as sqhole
import pygmsh
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #
def main():
    problemdata = createData()
    heat = simfempy.applications.heat.Heat(defgeom=createGeom, problemdata=problemdata)
    heat.static()
    simfempy.meshes.plotmesh.meshWithBoundaries(heat.mesh)
    result = heat.static()
    print(f"{result.info['timer']}")
    print(f"postproc: {result.data['global']['postproc']}")
    simfempy.meshes.plotmesh.meshWithData(heat.mesh, data=result.data, title="Heat example")
    plt.show()


# ---------------------------------------------------------------- #
def createGeom(h=0.1):
    lcar = h
    rect = [-1, 1, -1, 1]
    geom = pygmsh.built_in.Geometry()

    holes=[]
    holes.append(
        sqhole(geom, x=0.1, y=-0.4, r=0.3, lcar=lcar, label=200, make_surface=True)
    )
    holes.append(
        sqhole(geom, x=0.4, y=0.4, r=0.15, lcar=lcar, label=300, make_surface=True)
    )
    holes.append(
        sqhole(geom, x=-0.4, y=0.6, r=0.25, lcar=lcar, label=3000, make_surface=False)
    )
    p = geom.add_rectangle(*rect, z=0.0, lcar=0.1, holes=holes)
    geom.add_physical(p.surface, label=100)
    for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=1000 + i)
    return geom
    mesh = pygmsh.generate_mesh(geom)
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

# ---------------------------------------------------------------- #
def createData():
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.set("Robin", [1000])
    bdrycond.set("Dirichlet", [1001, 1003])
    bdrycond.set("Neumann", [1002, 3000, 3001, 3002, 3003])
    bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 0.01
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    bdrycond.fct[1000] = lambda x, y, z, nx, ny, nz: 100
    bdrycond.param[1000] = 100
    postproc = data.postproc
    postproc.type['bdrymean_low'] = "bdrymean"
    postproc.color['bdrymean_low'] = [1000]
    postproc.type['bdrymean_up'] = "bdrymean"
    postproc.color['bdrymean_up'] = [1002]
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = [1001, 1003]
    params = data.params
    params.set_scal_cells("kheat", [100], 0.001)
    params.set_scal_cells("kheat", [200, 300], 10.0)
    # alternative:
    # def kheat(label, x, y, z):
    #     if label==100: return 0.0001
    #     return 0.1*label
    # params.fct_glob["kheat"] = kheat
    return data


# ================================================================c#


main()