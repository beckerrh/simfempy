import matplotlib.pyplot as plt
from simfempy.meshes import testmeshes
from simfempy.meshes import plotmesh
from simfempy.tools.analyticalsolution import AnalyticalSolution
from simfempy.fems.rt0 import RT0
from simfempy.applications.transport import Transport
import simfempy.applications.problemdata

def plotBetaDownwind(betaC, beta, mesh):
    celldata = {f"beta": [betaC[:,i] for i in range(mesh.dimension)]}
    fig, axs = plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
    xd, ld = rt.downWind(beta)
    axs[0,0].plot(xd[:,0], xd[:,1], 'or')
    xd, ld = rt.downWind(beta, method='supg2')
    axs[0,0].plot(xd[:,0], xd[:,1], 'xb')
    plt.show()

#================================================================#
if __name__ == '__main__':
    mesh = testmeshes.unitsquare(0.5)
    plotmesh.plotmeshWithNumbering(mesh)
    plt.show()
    dim = mesh.dimension
    rt = RT0(mesh)
    # beta = rt.interpolate([AnalyticalSolution(expr="-y"),AnalyticalSolution(expr="x")])
    betafct = [AnalyticalSolution(expr="1"),AnalyticalSolution(expr="0")]
    beta = rt.interpolate(betafct)
    betaC = rt.toCell(beta)
    plotBetaDownwind(betaC, beta, mesh)
    data = simfempy.applications.problemdata.ProblemData()
    data.params.fct_glob['beta'] = betafct
    data.params.scal_glob['alpha'] = 0
    tr = Transport(mesh=mesh, problemdata=data, exactsolution="1+y+x")
    res = tr.static()
    plotmesh.meshWithData(mesh, data=res.data)
    plt.show()
    print(f"{res.info=}\n {res.data['global']}")
