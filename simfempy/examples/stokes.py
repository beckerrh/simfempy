assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.stokes import Stokes
from simfempy.applications.navierstokes import NavierStokes
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# ================================================================c#
def main():
    # create mesh and data
    mesh, data = drivenCavity()
    # mesh, data = backwardFacingStep()
    print(f"{mesh=}")
    # plotmesh.meshWithBoundaries(mesh)
    # create application
    # stokes = Stokes(mesh=mesh, problemdata=data)
    stokes = NavierStokes(mesh=mesh, problemdata=data)
    result = stokes.solve()
    print(f"{result.info['timer']}")
    print(f"postproc:")
    for p, v in result.data['global'].items(): print(f"{p}: {v}")
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
    plotmesh.meshWithBoundaries(mesh, fig=fig, outer=outer[0])
    plotmesh.meshWithData(mesh, data=result.data, title="Stokes", fig=fig, outer=outer[1])
    plotmesh.meshWithData(mesh, title="Stokes", fig=fig, outer=outer[2],
                          quiver_data={"V":list(result.data['point'].values())})
    plt.show()


# ================================================================c#
def drivenCavity(h=0.2):
    with pygmsh.geo.Geometry() as geom:
        ms = [h*v for v in [1.,1.,0.1,0.1]]
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=ms)
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
def backwardFacingStep(h=0.2):
    with pygmsh.geo.Geometry() as geom:
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003, 1005])
    data.bdrycond.set("Neumann", [1004])
    data.bdrycond.fct[1000] = lambda x, y, z: np.vstack((np.ones(x.shape[0]),np.zeros(x.shape[0])))
    # parameters
    data.params.scal_glob["mu"] = 1
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data

# ================================================================c#
main()