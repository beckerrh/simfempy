import matplotlib.pyplot as plt
from simfempy.meshes import testmeshes
from simfempy.meshes import plotmesh
from simfempy.tools.analyticalfunction import AnalyticalFunction
from simfempy.fems.rt0 import RT0
from simfempy.applications.transport import Transport
import simfempy.applications.problemdata

def plotBetaDownwind(betaC, beta, mesh, fem):
    import numpy as np
    celldata = {f"beta": [betaC[:,i] for i in range(mesh.dimension)]}
    fig, axs = plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
    xd, ld, delta = fem.downWind(beta)

    # xdl = np.einsum('nik,ni->nk', mesh.points[mesh.simplices],ld)
    # if not np.allclose(xd, xdl): raise ValueError(f"{xd=} {xdl=}")
    # xdiff = np.einsum('nik,ni->nk', mesh.points[mesh.simplices],ld-1/3)[:,:mesh.dimension]
    # betaloc = delta*betaC.T
    # if not np.allclose(betaloc.T, xdiff): raise ValueError(f"{betaloc=} {xdiff=}")

    axs[0,0].plot(xd[:,0], xd[:,1], 'or')
    xd, ld, delta = fem.downWind(beta, method='supg2')

    # xdl = np.einsum('nik,ni->nk', mesh.points[mesh.simplices], ld)
    # if not np.allclose(xd, xdl): raise ValueError(f"{xd=} {xdl=}")
    # xdiff = np.einsum('nik,ni->nk', mesh.points[mesh.simplices],ld-1/3)[:,:mesh.dimension]
    # betaloc = delta*betaC.T
    # if not np.allclose(betaloc.T, xdiff): raise ValueError(f"\n{betaloc=} \n{xdiff=}")

    axs[0,0].plot(xd[:,0], xd[:,1], 'xb')
    plt.show()

#================================================================#
if __name__ == '__main__':
    mesh = testmeshes.unitsquare(0.5)
    plotmesh.plotmeshWithNumbering(mesh)
    plt.show()
    dim = mesh.dimension
    # rt = RT0(mesh)
    # beta = rt.interpolate([AnalyticalFunction(expr="-y"),AnalyticalFunction(expr="x")])
    # betafct = [AnalyticalFunction(expr="1"),AnalyticalFunction(expr="0")]
    # beta = rt.interpolate(betafct)
    # betaC = rt.toCell(beta)
    data = simfempy.applications.problemdata.ProblemData()
    data.params.fct_glob['beta'] = ["y", "-x"]
    data.params.scal_glob['alpha'] = 0
    tr = Transport(mesh=mesh, problemdata=data, exactsolution="1+y+x", method="supg")
    res = tr.static()
    plotBetaDownwind(tr.betaC, tr.beta, mesh, tr.fem)
    plotmesh.meshWithData(mesh, data=res.data)
    plt.show()
    print(f"{res.info=}\n {res.data['global']}")
