import matplotlib.pyplot as plt
from simfempy.meshes import testmeshes
from simfempy.meshes import plotmesh
from simfempy.tools.analyticalsolution import AnalyticalSolution
from simfempy.fems.rt0 import RT0

#================================================================#
if __name__ == '__main__':
    mesh = testmeshes.unitsquare(2)
    dim = mesh.dimension
    rt = RT0(mesh)
    beta = rt.interpolate([AnalyticalSolution(expr="-y"),AnalyticalSolution(expr="x")])
    beta = rt.interpolate([AnalyticalSolution(expr="-7"),AnalyticalSolution(expr="11")])
    print(f"{beta.shape}")
    betaC = rt.toCell(beta)
    print(f"{betaC.shape}")
    # celldata = {f"beta{i}":betaC[i::dim] for i in range(dim)}
    celldata = {f"beta": [betaC[i::dim] for i in range(dim)]}
    plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
    plt.show()
