from unittest import TestLoader, TestSuite, TextTestRunner
import pylat.tests as tests
import pylat.expr_tests as expr_tests

if __name__ == "__main__":
    suite = TestSuite()
    loader = TestLoader()
    for t in tests.all_tests + expr_tests.all_tests:
        suite.addTest(loader.loadTestsFromTestCase(t))
    TextTestRunner(verbosity=2).run(suite)
