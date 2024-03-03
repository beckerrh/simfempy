import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

from simfempy.models.elliptic import Elliptic
from simfempy.models.application import Application
from simfempy.models.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh

# define Application class
class HeatExample(Application):
    def __init__(self):
        super().__init__(h=0.1)
        # fill problem data
        # boundary conditions
        self.problemdata.bdrycond.set("Dirichlet", [1000, 3000])
        self.problemdata.bdrycond.set("Neumann", [1001, 1002, 1003])
        self.problemdata.bdrycond.fct[1000] = lambda x, y, z: 200
        self.problemdata.bdrycond.fct[3000] = lambda x, y, z: 320
        # postprocess
        self.problemdata.postproc.set(name='bdrymean_right', type='bdry_mean', colors=1001)
        self.problemdata.postproc.set(name='bdrymean_left', type='bdry_mean', colors=1003)
        self.problemdata.postproc.set(name='bdrymean_up', type='bdry_mean', colors=1002)
        self.problemdata.postproc.set(name='bdrynflux', type='bdry_nflux', colors=[3000])
        # paramaters in equation
        self.problemdata.params.set_scal_cells("kheat", [100], 0.001)
        self.problemdata.params.set_scal_cells("kheat", [200], 10.0)
        # data.params.fct_glob["convection"] = ["0", "0.001"]
    def defineGeometry(self, geom, h):
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
disc_params={'dirichletmethod':'nitsche'}
heat = Elliptic(application=HeatExample(), fem='p1', disc_params=disc_params, linearsolver='pyamg')
heat.problemdata.params.fct_glob["initial_condition"] = "200"
t_final, dt, nframes = 5000, 100, 50
result = heat.dynamic(heat.initialCondition(), t_span=(0, t_final), nframes=nframes, dt=dt, theta=0.9)
# print(f"{result=}")

fig = plt.figure(constrained_layout=True)
fig.suptitle(f"{heat.application.__class__.__name__} (dynamic)", fontsize=16)
gs = fig.add_gridspec(2, 3)
nhalf = (nframes-1)//2
for i in range(3):
    # plotmesh.meshWithData(heat.mesh, data=data, alpha=0.5, fig=fig, outer=outer[1])
    data = heat.load_data(iter=i*nhalf)
    heat.application.plot(heat.mesh, data, fig=fig, gs=gs[i], title=f't={result.time[i*nhalf]}')
pp = heat.get_postprocs_dynamic()
ax = fig.add_subplot(gs[1, :])
for k,v in pp['postproc'].items():
    ax.plot(pp['time'], v, label=k)
ax.legend()
ax.grid()
plt.show()
# anim = animdata.AnimData(mesh, u)
# plt.show()
