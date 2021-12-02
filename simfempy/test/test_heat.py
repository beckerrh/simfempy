import testcaseanalytical
import numpy as np
import heat_analytic

#================================================================#
class TestAnalyticalHeat(testcaseanalytical.TestCaseAnalytical):
    def __init__(self, args):
        args['exactsolution'] = 'Linear'
        args['verbose'] = 0
        args['linearsolver'] = 'spsolve'
        super().__init__(args)
    def runTest(self):
        errors = heat_analytic.test(**self.args).errors
        self.checkerrors(errors)

#================================================================#
if __name__ == '__main__':
    from simfempy.tools import tools
    paramsdicts = {'dim':[1,2,3], 'fem':['p1','cr1'], 'dirichletmethod':['nitsche','strong']}
    testcaseanalytical.run(testcase=TestAnalyticalHeat, argss=tools.dictproduct(paramsdicts))
    # testcaseanalytical.run(testcase=TestAnalyticalHeat)
