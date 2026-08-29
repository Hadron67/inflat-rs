# Instructions

This is a computational package for lattice cosmology.

The subpackage `symlat/` uses Python as the interactive language and leverages LLVM to JIT-compile high-performance computational kernels. The repository root contains a `run_tests.py` script for running the tests under `symlat/`. The testing framework is `unittest`, not `pytest`.

For changes under `symlat/`, the scope part in the commit message should always be `symlat`, regardless of whether the changes are made in a subdirectory within `symlat/` or in the package root.
