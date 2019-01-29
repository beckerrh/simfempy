import unittest
import numpy as np

#================================================================#
class TestAnalytical(unittest.TestCase):
    def _check(self, results):
        for meth,err in results.items():
            if not np.all(err<1e-10): raise ValueError("error in method '{}' error is {}".format(meth,err))
#---------------------------
    def test_poisson2d(self):
        from heat.poisson import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_poisson3d(self):
        from heat.poisson import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))
    # ---------------------------
    def test_elliptic2d(self):
        from elliptic.elliptic import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_elliptic3d(self):
        from elliptic.elliptic import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))
    # ---------------------------
    def test_elasticity2d(self):
        from elasticity.analytic import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_elasticity3d(self):
        from elasticity.analytic import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))
    # ---------------------------
    def test_mixedlaplace2d(self):
        from mixed.laplace import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_mixedlaplace3d(self):
        from mixed.laplace import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))
    # ---------------------------
    def test_stokes2d(self):
        from flow.stokes import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_stokes3d(self):
        from flow.stokes import test_analytic
        self._check(test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))


#================================================================#
if __name__ == '__main__':
    unittest.main(verbosity=2)