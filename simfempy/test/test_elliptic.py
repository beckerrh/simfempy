import testcaseanalytical
import elliptic_analytic

#================================================================#
class TestAnalyticalElliptic(testcaseanalytical.TestCaseAnalytical):
    def __init__(self, args):
        args['exactsolution'] = 'Linear'
        args['verbose'] = 0
        args['linearsolver'] = 'spsolve'
        super().__init__(args)
    def runTest(self):
        errors = elliptic_analytic.test(**self.args).errors
        self.checkerrors(errors)

#================================================================#
from simfempy.tools import tools
paramsdicts = {'dim':[1,2,3], 'fem':['p1','cr1'], 'dirichletmethod':['nitsche','strong']}
testcaseanalytical.run(testcase=TestAnalyticalElliptic, argss=tools.dictproduct(paramsdicts))
