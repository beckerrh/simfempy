import unittest

#================================================================#
class TestHeat(unittest.TestCase):

    def test_poisson(self):
        import heat.poisson
        result = heat.poisson.test_analytic(problem = 'Analytic_Linear', geomname = "unitsquare", verbose=0)
        print("result", result)


#================================================================#
if __name__ == '__main__':
    unittest.main(verbosity=0)