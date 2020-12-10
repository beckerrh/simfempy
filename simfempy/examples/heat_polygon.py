import numpy as np
import matplotlib.pyplot as plt
import pygmsh
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh
from simfempy.applications.heat import Heat

# ---------------------------------------------------------------- #
def main():
    # create a mesh
    h = 0.1
    X = np.array([[0,-2], [1,-1], [1,2],[-1,2], [-1,-1]])
    X = np.insert(X, 2, 0, axis=1)
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon(X, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i,l in enumerate(p.lines):
            # print(f"adding {l} {1000+i}")
            geom.add_physical(l, label=f"{1000+i}")
        mesh = geom.generate_mesh()
    mesh =  SimplexMesh(mesh=mesh)
    plotmesh.meshWithBoundaries(mesh)
    plt.show()
    # create problem data
    data = ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.set("Dirichlet", [1000, 1004])
    bdrycond.set("Neumann", [1001, 1002, 1003])
    bdrycond.fct[1000] = lambda x,y,z: 200
    bdrycond.fct[1004] = lambda x,y,z: 300
    postproc = data.postproc
    postproc.type['bdrymean'] = "bdry_mean"
    postproc.color['bdrymean'] = [1002]
    params = data.params
    params.set_scal_cells("kheat", [100], 0.001)
# create heat application
    heat = Heat(mesh=mesh, problemdata=data)
    result = heat.static()
    print(f"{result.info['timer']}")
    print(f"postproc: {result.data['global']['postproc']}")
    plotmesh.meshWithData(heat.mesh, data=result.data, title="Heat static", alpha=1)
    plt.show()

# ---------------------------------------------------------------- #
main()