import unittest
import numpy as np
import heat_analytic
# import warnings
# warnings.simplefilter(action="error", category=DeprecationWarning)
#================================================================#
class TestAnalytical(unittest.TestCase):
    def checkerrors(self, errors, eps=1e-10):
        # print(f"{next(iter(errors.values())).keys()} {errors.keys()}")
        for meth,err in errors.items():
            assert isinstance(err, dict)
            for m, e in err.items():
                if not np.all(e<eps): raise ValueError("error in method '{}' '{}' error is {}".format(meth,m,e))
#================================================================#
class TestAnalyticalHeat(TestAnalytical):
    def __init__(self, args):
        super(TestAnalyticalHeat, self).__init__()
        self.args = args
    def runTest(self):
        args = self.args
        args['exactsolution'] = 'Linear'
        args['verbose'] = 0
        args['linearsolver'] = 'spsolve'
        errors = heat_analytic.test(**args).errors
        self.checkerrors(errors)

#================================================================#
if __name__ == '__main__':
    # unittest.main(verbosity=2)
    suite = unittest.TestSuite()
    paramsdicts = {'dim':[1,2,3], 'fem':['p1','cr1'], 'dirichletmethod':['nitsche','strong','new']}
    from simfempy.tools import tools
    for args in tools.dictproduct(paramsdicts):
        suite.addTest(TestAnalyticalHeat(args))
    unittest.TextTestRunner().run(suite)
