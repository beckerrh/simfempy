SIMple Finite Element Methods in PYthon

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib, sys
simfempypath = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0,simfempypath)

from simfempy.models.elliptic import Elliptic
from simfempy.models.application import Application
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
heat = Elliptic(application=HeatExample(), fem='p1', disc_params=disc_params, linearsolver={'method':'pyamg', 'disp':0, 'smoother':'schwarz'})
result, u = heat.static(mode="newton")
print(f"{result=}")
# for p, v in result.data['scalar'].items(): print(f"{p}: {v}")
fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"{heat.application.__class__.__name__} (static)", fontsize=16)
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
plotmesh.meshWithBoundaries(heat.mesh, fig=fig, outer=outer[0])
data = heat.sol_to_data(u)
data.update({'cell': {'k': heat.kheatcell}})
plotmesh.meshWithData(heat.mesh, data=data, alpha=0.5, fig=fig, outer=outer[1])
plt.show()
```
