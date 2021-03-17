import matplotlib.pyplot as plt
from simfempy.meshes import testmeshes
from simfempy.meshes import plotmesh
from simfempy.applications.transport import Transport
import simfempy.applications.problemdata
from simfempy.tools.comparemethods import CompareMethods

#-------------------------------------------------------------------------#
def plotBetaDownwind(betaC, beta, mesh, fem):
    import numpy as np
    celldata = {f"beta": [betaC[:,i] for i in range(mesh.dimension)]}
    fig, axs = plotmesh.meshWithData(mesh, quiver_cell_data=celldata, plotmesh=True)
    xd, ld, delta = fem.downWind(beta)
    axs[0,0].plot(xd[:,0], xd[:,1], 'or')
    xd, ld, delta = fem.downWind(beta, method='supg2')
    axs[0,0].plot(xd[:,0], xd[:,1], 'xb')
    plt.show()

def errors(appl, mesh):
    comp = CompareMethods(appl, plot=True)
    result = comp.compare(mesh=mesh)

#================================================================#
if __name__ == '__main__':
    mesh = testmeshes.unitsquare(0.5)
    plotmesh.plotmeshWithNumbering(mesh)
    plt.show()
    dim = mesh.dimension
    data = simfempy.applications.problemdata.ProblemData()
    data.params.fct_glob['beta'] = ["y", "-x"]
    data.params.fct_glob['beta'] = ["1", "1"]
    data.params.scal_glob['alpha'] = 0
    tr = Transport(mesh=mesh, problemdata=data, exactsolution="1+y+x", method="supg")
    tr.setMesh(mesh)
    plotBetaDownwind(tr.betaC, tr.beta, mesh, tr.fem)
    errors(tr, mesh)
    # res = tr.static()
    # plotmesh.meshWithData(mesh, data=res.data)
    # plt.show()
    # print(f"{res.info=}\n {res.data['global']}")
