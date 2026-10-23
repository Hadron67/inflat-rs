from unittest import TestLoader, TestSuite, TextTestRunner

from spy import tests as spy_tests
from symlat import tests as symlat_tests

if __name__ == "__main__":
    suite = TestSuite()
    loader = TestLoader()
    # tests.all_tests aggregates every test module (compile_test,
    # fn_wrapper_test, numpy_test and expr_tests)
    for t in symlat_tests.all_tests + spy_tests.all_tests:
        suite.addTest(loader.loadTestsFromTestCase(t))
    TextTestRunner(verbosity=2).run(suite)
