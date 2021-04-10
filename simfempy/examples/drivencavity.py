assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.stokesn import StokesN
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# ================================================================c#
def main():
    # create mesh and data
    mesh, data = drivenCavity()
    print(f"{mesh=}")
    # plotmesh.meshWithBoundaries(mesh)
    # create application
    stokes = StokesN(mesh=mesh, problemdata=data)
    result = stokes.static()
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
def drivenCavity(h=0.2):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
    data.bdrycond.fct[1002] = lambda x, y, z: np.vstack((np.ones(x.shape[0]),np.zeros(x.shape[0])))
    # parameters
    data.params.scal_glob["mu"] = 1
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data

# ================================================================c#
main()
