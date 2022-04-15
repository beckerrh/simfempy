import numpy as np

assert __name__ == '__main__'
import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,simfempypath)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import simfempy.meshes.testmeshes as testmeshes
from simfempy.applications.beam import Beam
import simfempy.applications.problemdata
from simfempy.meshes import plotmesh

# ================================================================c#
mesh = testmeshes.unitline(h=0.2)
data = simfempy.applications.problemdata.ProblemData()
data.bdrycond.set("Clamped", [10000])
# data.bdrycond.set("SimplySupported", [10001])
data.bdrycond.set("Forces", [10001])
# data.bdrycond.fct[10001] = lambda x, y, z, nx, ny, nz: 0, lambda x, y, z, nx, ny, nz: 2
data.params.fct_glob['rhs'] = lambda x, y, z: np.full_like(x, 1)
data.params.scal_glob["EI"] = 1
beam = Beam(mesh=mesh, problemdata=data)
result = beam.static()
print(f"postproc:")
for p, v in result.data['global'].items(): print(f"{p}: {v}")
fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
# plotmesh.meshWithBoundaries(mesh, fig=fig, outer=outer[0])
plotmesh.meshWithData(mesh, data=result.data, title="Beam", alpha=1, fig=fig, outer=outer[1])
# plotmesh.meshWithData(mesh, data=result.data, title="Elast", alpha=1)
plt.show()
