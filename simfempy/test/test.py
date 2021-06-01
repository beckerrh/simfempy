import unittest
import numpy as np

#================================================================#
class TestAnalytical(unittest.TestCase):
    def _check(self, results, eps=1e-10):
        for meth,err in results.items():
            if isinstance(err, dict):
                for m, e in err.items():
                    if not np.all(e<eps): raise ValueError("error in method '{}' '{}' error is {}".format(meth,m,e))
            else:
                if not np.all(err<eps): raise ValueError("error in method '{}' error is {}".format(meth,err))
#---------------------------
    def test_poisson1d(self):
        from heat_analytic import test
        self._check(test(dim=1, exactsolution = 'Linear', verbose=0, linearsolver='umf'))
    def test_poisson2d(self):
        from heat_analytic import test
        self._check(test(dim=2, exactsolution = 'Linear', verbose=0))
    def test_poisson3d(self):
        from heat_analytic import test
        self._check(test(dim=3, exactsolution = 'Linear', verbose=0, linearsolver='umf'))
    # ---------------------------
    def test_elasticity2d(self):
        from elasticity_analytic import test
        self._check(test(dim=2, exactsolution = 'Linear', linearsolver='umf', verbose=0))
    def test_elasticity3d(self):
        from elasticity_analytic import test
        self._check(test(dim=3, exactsolution = 'Linear', linearsolver='umf', verbose=0, niter=2))
    # ---------------------------
    def test_stokes2d(self):
        from stokes_analytic import test
        self._check(test(dim=2, exactsolution='Linear', linearsolver='umf', verbose=0))
    def test_stokes3d(self):
        from stokes_analytic import test
        self._check(test(dim=3, exactsolution='Linear', linearsolver='umf', verbose=0))


#     # ---------------------------
#     def test_mixedlaplace2d(self):
#         from mixed.laplace import test_analytic
#         self._check(test_analytic(exactsolution = 'Linear', geomname = "unitsquare", verbose=0))
#     def test_mixedlaplace3d(self):
#         from mixed.laplace import test_analytic
#         self._check(test_analytic(exactsolution = 'Linear', geomname = "unitcube", verbose=0))
#     # ---------------------------
#     def test_stokes2d(self):
#         from flow.stokes import test_analytic
#         self._check(test_analytic(exactsolution = 'Linear', geomname = "unitsquare", verbose=0))
#     def test_stokes3d(self):
#         from flow.stokes import test_analytic
#         self._check(test_analytic(exactsolution = 'Linear', geomname = "unitcube", verbose=0))


#================================================================#
if __name__ == '__main__':
    unittest.main(verbosity=2)