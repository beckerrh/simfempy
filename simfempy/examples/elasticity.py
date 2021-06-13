assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.elasticity import Elasticity
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# ================================================================c#
def main():
    # create mesh
    mesh = createMesh()
    print(f"{mesh=}")
    # plotmesh.meshWithBoundaries(mesh)
    # create problem data
    data = createProblemData()
    # create application
    elasticity = Elasticity(mesh=mesh, problemdata=data)
    result = elasticity.static()
    print(f"{result.info['timer']}")
    print(f"postproc:")
    for p, v in result.data['global'].items(): print(f"{p}: {v}")
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    plotmesh.meshWithBoundaries(mesh, fig=fig, outer=outer[0])
    # plotmesh.meshWithData(mesh, data=result.data, title="Elast", alpha=1, fig=fig, outer=outer[1])
    plotmesh.meshWithData(mesh, data=result.data, title="Elast", alpha=1)
    plt.show()


# ================================================================c#
def createMesh(h=0.2):
    with pygmsh.geo.Geometry() as geom:
        basepoints = [ [+0.0, +0.5], [-0.1, +0.1], [-0.5, +0.0], [-0.1, -0.1],
                [+0.0, -0.5], [+0.1, -0.1], [+0.5, +0.0], [+0.1, +0.1] ]
        poly = geom.add_polygon(basepoints, mesh_size=h,)
        geom.add_physical(poly.surface, label="100")
        top, vol, ext = geom.twist(
            poly,
            translation_axis=[0, 0, 1], rotation_axis=[0, 0, 1], point_on_axis=[0, 0, 0], angle=np.pi / 3,
        )
        geom.add_physical(top, label=f"{102}")
        geom.add_physical(ext, label=f"{101}")
        geom.add_physical(vol, label="10")
        mesh = geom.generate_mesh()
    return SimplexMesh(mesh=mesh)

# ================================================================c#
def createProblemData():
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [100, 102])
    data.bdrycond.set("Neumann", [101])
    data.bdrycond.fct[102] = lambda x, y, z: np.vstack((np.ones(x.shape),-np.ones(x.shape)))
    # postprocess
    # data.postproc.set(name='bdrymean_right', type='bdry_mean', colors=1001)
    # paramaters in equation
    # data.params.fct_glob["convection"] = ["0", "0.001"]
    return data

# ================================================================c#
main()
