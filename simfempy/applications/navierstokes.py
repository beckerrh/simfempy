from simfempy.applications.stokes import Stokes

class NavierStokes(Stokes):
    def solve(self, iter=100, dirname='Run'):
        return self.static(iter, dirname, mode='nonlinear')
    def computeDefect(self, u):
        return self.A@u-self.b
    def computeDx(self, b, u):
        try:
            u, niter = self.linearSolver(self.A, b, u, solver=self.linearsolver)
        except Warning:
            raise ValueError(f"matrix is singular {self.A.shape=} {self.A.diagonal()=}")
        self.timer.add('solve')
        return u
