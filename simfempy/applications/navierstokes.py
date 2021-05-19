import numpy as np
from simfempy.applications.stokes import Stokes

class NavierStokes(Stokes):
    def solve(self, iter=100, dirname='Run'):
        return self.static(iter, dirname, mode='nonlinear')
