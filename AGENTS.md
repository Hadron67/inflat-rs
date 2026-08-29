# Instructions

This is a computational package for lattice cosmology.

## Key Directories and Files
* `symlat/` uses Python as the interactive language and leverages LLVM to JIT-compile high-performance computational kernels.
* `lat/` implements lattice simulations for various models, such as single-field inflation.
* The repository root contains a `run_tests.py` script for running the tests under `symlat/`. The testing framework is `unittest`, not `pytest`.

For changes under any of these directories, the scope part in the commit message should always be the name of that directory, regardless of whether the changes are made in a subdirectory within or in the package root. For example, after modifying `symlat/jit/compile.py`, the scope should be `symlat`.

## Coding Guidelines

* In Python sources, prefer relative imports.
