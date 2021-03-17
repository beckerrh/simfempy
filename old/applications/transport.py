import numpy as np
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.fems.rt0 import RT0
from simfempy.tools.analyticalfunction import AnalyticalFunction

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
        beta_given = self.problemdata.params.fct_glob['beta']
        if not isinstance(beta_given,list):
            p = "problemdata.params.fct_glob['beta']"
            raise ValueError(f"need '{p}' as a list of length dim of str or AnalyticalSolution")
        elif isinstance(beta_given[0],str):
            self.problemdata.params.fct_glob['beta'] = [AnalyticalFunction(expr=e) for e in beta_given]
        self.linearsolver = kwargs.pop('linearsolver','umf')
        fem = kwargs.pop('fem','p1')
        if fem == 'p1':
            self.fem = fems.p1.P1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="supg"
    def _checkProblemData(self):
        pass
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            alpha = self.problemdata.params.scal_glob['alpha']
            beta = self.problemdata.params.fct_glob['beta']
            rhs = alpha*solexact(x,y,z)
            for i in range(self.mesh.dimension):
                rhs += beta[i](x,y,z) * solexact.d(i, x, y, z)
            return rhs
        return _fctu

    def defineBdryFctAnalyticalSolution(self, color):
        return self.dirichletfct()
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self._checkProblemData()
        self.fem.setMesh(self.mesh)
        betafct = self.problemdata.params.fct_glob['beta']
        rt = RT0(mesh)
        self.beta = rt.interpolate(betafct)
        self.xd, self.lamdbeta, self.delta = self.fem.downWind(self.beta, method=self.method)
        self.betaC = rt.toCell(self.beta)
    def computeMatrix(self):
        A =  self.fem.comptuteMatrixTransport(self.beta, self.betaC, self.lamdbeta)
        # print(f"{self.fem.nunknowns()=} {A.data=}")
        return A
    def computeRhs(self, u=None):
        b = np.zeros(self.fem.nunknowns())
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolate(self.problemdata.params.fct_glob['rhs'])
            # print(f"{fp1=}")
            # A = self.fem.computeMassMatrixSupg(self.xd)
            # b += A.dot(fp1)
            self.fem.massDot(b, fp1)
            self.fem.massDotSupg(b, fp1, self.xd)
        if self.problemdata.solexact:
            f = self.fem.interpolate(self.problemdata.solexact)
        self.fem.massDotBoundary(b, f, coeff=-np.minimum(self.beta,0))
        # print(f"{b=}")
        return b,u
    def postProcess(self, u):
        data = {'point':{}, 'global':{}}
        data['point']['U'] = self.fem.tonode(u)
        if self.problemdata.solexact:
            data['cell'] = {}
            data['global']['pcL2'], ec = self.fem.computeErrorL2Cell(self.problemdata.solexact, u)
            data['global']['pnL2'], en = self.fem.computeErrorL2Node(self.problemdata.solexact, u)
            data['cell']['err'] = ec
        return data


#=================================================================#
if __name__ == '__main__':
    print("Pas de test")
