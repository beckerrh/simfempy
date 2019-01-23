import unittest
import numpy as np

#================================================================#
class TestAnalytical(unittest.TestCase):
    def _check(self, result):
        for meth,err in result.items():
            if not np.all(err<1e-10): raise ValueError("error in method '{}' error is {}".format(meth,err))
    def test_poisson2d(self):
        import heat.poisson
        self._check(heat.poisson.test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_poisson3d(self):
        import heat.poisson
        self._check(heat.poisson.test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))
    def test_elliptic2d(self):
        import elliptic.elliptic
        self._check(elliptic.elliptic.test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_elliptic3d(self):
        import elliptic.elliptic
        self._check(elliptic.elliptic.test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))
    def test_elasticity2d(self):
        import elasticity.analytic
        self._check(elasticity.analytic.test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0))
    def test_elasticity3d(self):
        import elasticity.analytic
        self._check(elasticity.analytic.test_analytic(problem = 'Analytic_Linear', geomname = "unitcube", verbose=0))


#================================================================#
if __name__ == '__main__':
    unittest.main(verbosity=2)