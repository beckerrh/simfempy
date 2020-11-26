import matplotlib.pyplot as plt
from simfempy.meshes import testmeshes
from simfempy.meshes import plotmesh
from simfempy.tools.analyticalsolution import AnalyticalSolution
from simfempy.fems.rt0 import RT0
from simfempy.fems.p1 import P1
from simfempy.applications.transport import Transport

#================================================================#
if __name__ == '__main__':
    mesh = testmeshes.unitsquare(1.5)
    dim = mesh.dimension
    rt = RT0(mesh)
    # beta = rt.interpolate([AnalyticalSolution(expr="-y"),AnalyticalSolution(expr="x")])
    beta = rt.interpolate([AnalyticalSolution(expr="-1"),AnalyticalSolution(expr="1")])
    betaC = rt.toCell(beta)
    # celldata = {f"beta{i}":betaC[i::dim] for i in range(dim)}
    celldata = {f"beta": [betaC[:,i] for i in range(dim)]}
    fig, axs = plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
    xd, ld = rt.downWind(beta)
    axs[0,0].plot(xd[:,0], xd[:,1], 'xr')
    xd, ld = rt.downWind(beta, method='supg2')
    axs[0,0].plot(xd[:,0], xd[:,1], 'xb')

    tr = Transport(mesh=mesh)
