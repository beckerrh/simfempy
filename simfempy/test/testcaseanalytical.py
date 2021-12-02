import unittest
import numpy as np
# import warnings
# warnings.simplefilter(action="error", category=DeprecationWarning)
#================================================================#
class TestCaseAnalytical(unittest.TestCase):
    failed = []
    def __init__(self, args):
        super().__init__()
        self.args = args
    def checkerrors(self, errors, eps=1e-10):
        # print(f"{next(iter(errors.values())).keys()} {errors.keys()}")
        failed = {}
        for meth,err in errors.items():
            assert isinstance(err, dict)
            for m, e in err.items():
                if not np.all(e < eps): failed[meth] = e
        if len(failed):
            TestCaseAnalytical.failed.append(self.args)
            self.fail(msg=f'Test case failed {self.args=}\n{failed=}')
# class TestResult(unittest.TestResult):
#     def addFailure(self, test, err):
#         print(f'#### test case failed {test.args=}')
#         super().addFailure(test, err)
#     def addError(self, test, err):
#         print(f'#### test case error {dir(test)=}')
#         super().addError(test, err)
#

#================================================================#
def run(testcase, argss=None):
    import json
    filename = f"{testcase.__name__}_Failed.txt"
    suite = unittest.TestSuite()
    if not argss:
        with open(filename, 'r') as f:
            argss = json.loads(f.read())
    for args in argss:
        suite.addTest(testcase(args))
    # unittest.TextTestRunner(resultclass=TestResult).run(suite)
    unittest.TextTestRunner().run(suite)
    with open(filename, 'w') as f:
        f.write(json.dumps(TestCaseAnalytical.failed))
    # print(f"{TestCaseAnalytical.failed=}")
