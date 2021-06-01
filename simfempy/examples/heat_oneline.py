assert __name__ == '__main__'
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.heat import Heat
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# create mesh
h=0.1
with pygmsh.geo.Geometry() as geom:
    holes = []
    rectangle = geom.add_rectangle(xmin=-1.5, xmax=-0.5, ymin=-1.5, ymax=-0.5, z=0, mesh_size=h)
    geom.add_physical(rectangle.surface, label="200")
    geom.add_physical(rectangle.lines, label="20")  # required for correct boundary labels (!?)
    holes.append(rectangle)
    circle = geom.add_circle(x0=[0, 0], radius=0.5, mesh_size=h, num_sections=6, make_surface=False)
    geom.add_physical(circle.curve_loop.curves, label="3000")
    holes.append(circle)
    p = geom.add_rectangle(xmin=-2, xmax=2, ymin=-2, ymax=2, z=0, mesh_size=h, holes=holes)
    geom.add_physical(p.surface, label="100")
    for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
    mesh = geom.generate_mesh()
mesh = SimplexMesh(mesh=mesh)
# create problem data
data = ProblemData()
# boundary conditions
data.bdrycond.set("Dirichlet", [1000, 3000])
data.bdrycond.set("Neumann", [1001, 1002, 1003])
data.bdrycond.fct[1000] = lambda x, y, z: 200
data.bdrycond.fct[3000] = lambda x, y, z: 320
# postprocess
data.postproc.set(name='bdrymean_right', type='bdry_mean', colors=1001)
data.postproc.set(name='bdrymean_left', type='bdry_mean', colors=1003)
data.postproc.set(name='bdrymean_up', type='bdry_mean', colors=1002)
data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=[3000])
# paramaters in equation
data.params.set_scal_cells("kheat", [100], 0.001)
data.params.set_scal_cells("kheat", [200], 10.0)
data.params.fct_glob["convection"] = ["0", "0.001"]
# create application
heat = Heat(mesh=mesh, problemdata=data, fem='p1')
result = heat.static()
print(f"{result.info['timer']}")
print(f"postproc:")
for p, v in result.data['global'].items(): print(f"{p}: {v}")
fig = plt.figure(figsize=(10, 8))
fig.suptitle("Heat static", fontsize=16)
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
plotmesh.meshWithBoundaries(heat.mesh, fig=fig, outer=outer[0])
plotmesh.meshWithData(heat.mesh, data=result.data, alpha=0.5, fig=fig, outer=outer[1])
plt.show()
