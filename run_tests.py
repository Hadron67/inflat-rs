from unittest import TestLoader, TestSuite, TextTestRunner
import pylat.tests as tests

if __name__ == "__main__":
    suite = TestSuite()
    loader = TestLoader()
    for t in tests.all_tests:
        suite.addTest(loader.loadTestsFromTestCase(t))
    TextTestRunner(verbosity=2).run(suite)
