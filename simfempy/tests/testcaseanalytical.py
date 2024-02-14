import unittest
import numpy as np
# import warnings
# warnings.simplefilter(action="error", category=DeprecationWarning)
#================================================================#
class TestCaseAnalytical(unittest.TestCase):
    # static variable
    failed = []
    def __init__(self, test, eps=1e-10):
        self.eps = eps
        super().__init__()
        self.test = test
    def checkerrors(self, errors):
        eps = self.eps
        # print(f"{next(iter(errors.values())).keys()} {errors.keys()}")
        failed_met = {}
        for meth,err in errors.items():
            assert isinstance(err, dict)
            for m, e in err.items():
                if not np.all(e < eps): failed_met[meth] = e
        if len(failed_met):
            self.failed.append(self.test)
            self.fail(msg=f'Test case failed {self.test=}\n{failed_met=}')

#================================================================#
def run(testcase, tests=None):
    import os, json
    filename = f"{testcase.__name__}_Failed.txt"
    if os.path.exists(filename) and os.path.getsize(filename) > 2:
        # check if log-file exists and contains more than empty list
        with open(filename, 'r') as f:
            tests = json.loads(f.read())
    suite = unittest.TestSuite()
    # static variable
    TestCaseAnalytical.failed = []
    for test in tests:
        suite.addTest(testcase(test))
    unittest.TextTestRunner().run(suite)
    with open(filename, 'w') as f:
        f.write(json.dumps(TestCaseAnalytical.failed))
