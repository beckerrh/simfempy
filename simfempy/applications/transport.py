import numpy as np
from simfempy import fems
from simfempy.applications.application import Application

#=================================================================#
class Transport(Application):
    """
    Class for the (stationary) transport equation
    $$
    alpha u + \div(beta u) = f   domain
    beta\cdot n = g              bdry
    $$
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the constructor:
        fem: only p1 or cr1
        problemdata
        method
        plotk
    Paramaters used from problemdata:
        alpha : global constant from problemdata.paramglobal
        beta : RT0 field
    Possible parameters for computaion of postprocess:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'linearsolver' in kwargs: self.linearsolver = kwargs.pop('linearsolver')
        else: self.linearsolver = 'pyamg'
        fem = 'p1'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        if fem == 'p1':
            self.fem = fems.p1.P1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="supg"
    def _checkProblemData(self):
        self.problemdata.check(self.mesh)
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            alpha = self.problemdata.params.scal_glob['alpha']
            beta = self.problemdata.params.scal_glob['beta']
            rhs = alpha*solexact(x,y,z)
            for i in range(self.mesh.dimension):
                rhs += beta[i] * solexact.d(i, x, y, z)
            return rhs
        return _fctu
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self._checkProblemData()
        self.fem.setMesh(self.mesh)
    def computeMatrix(self):
        return A
    def computeRhs(self, u=None):
        return b,u
    def postProcess(self, u):
        point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        return point_data, side_data, cell_data, global_data


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
