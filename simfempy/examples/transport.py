assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.heat import Heat
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# ---------------------------------------------------------------- #
def main(h):
    #create mesh
    mesh, problemdata = ramp(h=h)
    #create application
    heat = Heat(mesh=mesh, problemdata=problemdata, fem='cr1', stab='upw', dirichletmethod='nitsche', masslumpedbdry=False)
    # heat = Heat(mesh=mesh, problemdata=problemdata, fem='p1', stab='upw2', dirichletmethod='strong', masslumpedbdry=True)
    # heat.fem.plotBetaDownwind()
    # return
    # result = heat.static(mode='nonlinear')
    result = heat.static()
    # print(f"{heat=}")
    # print(f"postproc:")
    for p,v in result.data['global'].items(): print(f"{p}: {v}")
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    plotmesh.meshWithData(mesh, point_data=result.data['point'], title="Ramp", alpha=1,fig=fig, outer=outer[0])
    # plotmesh.meshTriSurf(mesh, data=result.data['point']['U'], title="Ramp", alpha=1,fig=fig, outer=outer[1])
    uh = result.data['point']['U']
    print(f"{np.min(uh)=} {np.max(uh)=}")
    plotmesh.meshTriShading(mesh, data=uh, title="Ramp", alpha=1,fig=fig, outer=outer[1])
    plt.show()

# ---------------------------------------------------------------- #
def ramp(h=0.2):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(xmin=-1, xmax=1, ymin=-1, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    mesh = SimplexMesh(mesh=mesh)
    data = ProblemData()
    #boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1003,1001, 1002])
    data.bdrycond.set("Neumann", [])
    data.bdrycond.fct[1000] = lambda x,y,z: x>0
    #postprocess
    data.postproc.set(name='bdrymean_right', type='bdry_mean', colors=1001)
    data.postproc.set(name='bdrymean_left', type='bdry_mean', colors=1003)
    data.postproc.set(name='bdrymean_up', type='bdry_mean', colors=1002)
    #paramaters in equation
    data.params.scal_glob["kheat"] = 0
    data.params.fct_glob["convection"] = ["0.5", "1"]
    return mesh, data

# ================================================================c#

# main(mode='static', convection=True)
main(h=1.1)