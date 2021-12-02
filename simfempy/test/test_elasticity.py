import testcaseanalytical
import numpy as np
import elasticity_analytic

#================================================================#
class TestAnalyticalElasticity(testcaseanalytical.TestCaseAnalytical):
    def __init__(self, args):
        args['exactsolution'] = 'Linear'
        args['verbose'] = 0
        args['linearsolver'] = 'spsolve'
        super().__init__(args)
    def runTest(self):
        errors = elasticity_analytic.test(**self.args).errors
        self.checkerrors(errors)

#================================================================#
from simfempy.tools import tools
paramsdicts = {'dim':[2,3], 'fem':['p1','cr1'], 'dirichletmethod':['strong']}
testcaseanalytical.run(testcase=TestAnalyticalElasticity, argss=tools.dictproduct(paramsdicts))
