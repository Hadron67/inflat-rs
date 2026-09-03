"""The ``symlat`` test suites.

The test classes have been split by content into
:mod:`symlat.jit.compile_test`, :mod:`symlat.jit.fn_wrapper_test` and
:mod:`symlat.jit.numpy_test`.  This module re-exports them so that
``run_tests.py`` and existing imports of ``symlat.tests`` keep working.
"""

from .expr_tests import all_tests as expr_tests
from .jit.compile_test import all_tests as compile_tests
from .jit.fn_wrapper_test import all_tests as fn_wrapper_tests
from .jit.numpy_test import all_tests as numpy_tests

all_tests = compile_tests + fn_wrapper_tests + numpy_tests + expr_tests
