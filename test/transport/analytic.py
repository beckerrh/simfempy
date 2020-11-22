import matplotlib.pyplot as plt
from simfempy.meshes import testmeshes
from simfempy.meshes import plotmesh
from simfempy.tools.analyticalsolution import AnalyticalSolution
from simfempy.fems.rt0 import RT0
from simfempy.fems.p1 import P1

#================================================================#
if __name__ == '__main__':
    mesh = testmeshes.unitsquare(0.5)
    dim = mesh.dimension
    rt = RT0(mesh)
    # beta = rt.interpolate([AnalyticalSolution(expr="-y"),AnalyticalSolution(expr="x")])
    beta = rt.interpolate([AnalyticalSolution(expr="-1"),AnalyticalSolution(expr="1")])
    print(f"{beta.shape}")
    betaC = rt.toCell(beta)
    print(f"{betaC.shape}")
    # celldata = {f"beta{i}":betaC[i::dim] for i in range(dim)}
    celldata = {f"beta": [betaC[:,i] for i in range(dim)]}
    fig, axs = plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
    xd = rt.downWind(beta)
    axs[0,0].plot(xd[:,0], xd[:,1], 'xr')
    p1 = P1(mesh)
    A = p1.comptuteMatrixTransport(beta, betaC, xd)
    plt.show()
