import pathlib, sys
simfempypath = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0,simfempypath)

import simfempy.tests.testcaseanalytical
import stokes_analytic

#================================================================#
class TestAnalyticalStokes(simfempy.tests.testcaseanalytical.TestCaseAnalytical):
    def __init__(self, test):
        self.args = {}
        modelargs = {'stack_storage': False}
        modelargs['singleA'] = True
        modelargs['mode'] = 'newton'
        self.args['modelargs'] = modelargs
        self.args['exactsolution'] = 'Linear'
        self.args['verbose'] = 0
        self.args['linearsolver'] = 'spsolve'
        self.args.update(test)
        super().__init__(test, eps=1e-8)
    def runTest(self):
        errors = stokes_analytic.test(**self.args).errors
        self.checkerrors(errors)

#================================================================#
tests = [{'dim':2}, {'dim':3}]
simfempy.tests.testcaseanalytical.run(testcase=TestAnalyticalStokes, tests=tests)
