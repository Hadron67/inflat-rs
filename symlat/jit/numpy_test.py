from typing import cast
from unittest import TestCase

import numpy as np
from llvmlite import binding as llvm
from numpy.testing import assert_almost_equal

from .numpy import ArrayNode, ArrayWrapper, JitContext
from .openmp import OpenMPBackend

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class NumpyJitTest(TestCase):
    """Tests for the lazy numpy-style frontend in ``symlat.jit.numpy``."""

    @staticmethod
    def _data(w) -> np.ndarray:
        """The concrete numpy array behind a wrapper (leaf or computed)."""
        return cast(ArrayNode, w.arr).arr

    def test_usage_example(self):
        np.random.seed(114514)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(8, 9, 10)
        b = nc.rand(8, 9, 10)

        c = a + b  # lazy: no computation happens yet

        d = nc.zeros(*a.shape)
        d[...] = c  # compiles and runs `c` into `d`

        assert_almost_equal(self._data(d), self._data(a) + self._data(b))

    def test_lazy_arithmetic(self):
        np.random.seed(1)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)

        d = nc.zeros(4, 5)
        d[...] = a * b + a - b / 2 + 2 * a ** 2

        expected = self._data(a) * self._data(b) + self._data(a) - self._data(b) / 2 + 2 * self._data(a) ** 2
        assert_almost_equal(self._data(d), expected)

    def test_broadcast(self):
        nc = JitContext(OpenMPBackend())

        np.random.seed(2)
        a = nc.rand(4, 5)
        b = nc.rand(5)  # rank-1, trailing-aligned with the last axis

        d = nc.zeros(4, 5)
        d[...] = a + b
        assert_almost_equal(self._data(d), self._data(a) + self._data(b))

    def test_in_place(self):
        np.random.seed(3)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        a0 = self._data(a).copy()

        a[...] = a + b
        assert_almost_equal(self._data(a), a0 + self._data(b))

    def test_scalar_assignment(self):
        nc = JitContext(OpenMPBackend())

        d = nc.zeros(4, 5)
        d[...] = 3.0
        assert_almost_equal(self._data(d), np.full((4, 5), 3.0))

    def test_cache(self):
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        d = nc.zeros(4, 5)

        d[...] = a + b
        d[...] = a + b  # the same assignment must reuse the compiled kernel
        assert len(nc._cache) == 1
        assert_almost_equal(self._data(d), self._data(a) + self._data(b))

    def test_cache_shares_identical_structure(self):
        # the kernel is keyed by the expression structure and the argument
        # dtypes/ranks, so structurally identical assignments share it even for
        # different arrays
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        c = nc.rand(4, 5)
        d1 = nc.zeros(4, 5)
        d2 = nc.zeros(4, 5)

        d1[...] = a + b
        d2[...] = b + c
        assert len(nc._cache) == 1
        assert_almost_equal(self._data(d1), self._data(a) + self._data(b))
        assert_almost_equal(self._data(d2), self._data(b) + self._data(c))

    def test_array_node_is_not_comparable(self):
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        with self.assertRaises(NotImplementedError):
            a.arr.compare(b.arr)

    def test_slice_assignment(self):
        np.random.seed(6)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(5)
        c = nc.rand(4)
        expected = self._data(a).copy()

        a[0] = b
        expected[0] = self._data(b)
        a[:, 3] = c
        expected[:, 3] = self._data(c)
        assert_almost_equal(self._data(a), expected)

    def test_slice_assignment_negative_index(self):
        np.random.seed(7)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(5)
        expected = self._data(a).copy()

        a[-1] = b
        expected[-1] = self._data(b)
        assert_almost_equal(self._data(a), expected)

    def test_slice_assignment_scalar_and_element(self):
        nc = JitContext(OpenMPBackend())

        a = nc.zeros(4, 5)
        a[1] = 2.5
        a[2, 4] = 7.0
        expected = np.zeros((4, 5))
        expected[1] = 2.5
        expected[2, 4] = 7.0
        assert_almost_equal(self._data(a), expected)

    def test_slice_assignment_in_place(self):
        np.random.seed(8)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        expected = self._data(a).copy()

        a[0] = a[1] + b[2]
        expected[0] = expected[1] + self._data(b)[2]
        assert_almost_equal(self._data(a), expected)

    def test_slice_read(self):
        np.random.seed(9)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        d = nc.zeros(4, 5)

        d[0] = a[1] * 2 + b[3]
        d[:, 2] = a[:, 3]
        expected = np.zeros((4, 5))
        expected[0] = self._data(a)[1] * 2 + self._data(b)[3]
        expected[:, 2] = self._data(a)[:, 3]
        assert_almost_equal(self._data(d), expected)

    def test_slice_assignment_errors(self):
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 6)

        with self.assertRaises(ValueError):
            a[0] = b  # cannot broadcast (4, 6) into (5,)

        with self.assertRaises(TypeError):
            a[0:2] = b  # range slices are not supported

        with self.assertRaises(TypeError):
            a[..., 0] = b  # ellipsis is not supported as an index

        with self.assertRaises(IndexError):
            a[10] = nc.zeros(5)  # index out of bounds

    def test_errors(self):
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(6, 7)
        d = nc.zeros(4, 5)

        with self.assertRaises(ValueError):
            d[...] = b  # incompatible shapes

        c = a + b  # a computed expression has no storage of its own
        with self.assertRaises(TypeError):
            c[...] = a
        with self.assertRaises(TypeError):
            c[0] = a  # cannot slice-assign into a computed expression

        with self.assertRaises(TypeError):
            d[...] = self._data(a)  # raw numpy arrays are not operands

    def test_sum(self):
        np.random.seed(10)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        result = nc.sum(a)
        assert_almost_equal(result, np.sum(self._data(a)))
        # sum is eager: it returns a plain numpy scalar, not a wrapper
        self.assertIsInstance(result, np.generic)

    def test_sum_of_expression(self):
        np.random.seed(11)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        assert_almost_equal(
            nc.sum(a * b + a - 2),
            np.sum(self._data(a) * self._data(b) + self._data(a) - 2),
        )

    def test_sum_broadcast(self):
        np.random.seed(12)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(5)
        assert_almost_equal(nc.sum(a + b), np.sum(self._data(a) + self._data(b)))

    def test_sum_slices(self):
        np.random.seed(13)
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        assert_almost_equal(nc.sum(a[0]), np.sum(self._data(a)[0]))
        assert_almost_equal(nc.sum(a[:, 2]), np.sum(self._data(a)[:, 2]))
        assert_almost_equal(nc.sum(a[-1]), np.sum(self._data(a)[-1]))

    def test_sum_complex_and_int(self):
        nc = JitContext(OpenMPBackend())

        complex_arr = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
        assert_almost_equal(
            nc.sum(ArrayWrapper(nc, ArrayNode(complex_arr))), np.sum(complex_arr)
        )
        int_arr = np.arange(12).reshape(3, 4)
        assert_almost_equal(nc.sum(ArrayWrapper(nc, ArrayNode(int_arr))), np.sum(int_arr))

    def test_sum_cache(self):
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        b = nc.rand(4, 5)
        c = nc.rand(4, 5)
        nc.sum(a + b)
        nc.sum(a + b)  # the same expression structure reuses the compiled kernel
        nc.sum(b + c)
        self.assertEqual(len(nc._reduction_cache), 1)
        nc.sum(a * b)  # a different structure compiles separately
        self.assertEqual(len(nc._reduction_cache), 2)

    def test_sum_errors(self):
        nc = JitContext(OpenMPBackend())

        a = nc.rand(4, 5)
        with self.assertRaises(TypeError):
            nc.sum(a, axis=0)  # only the sum over all axes is supported
        with self.assertRaises(TypeError):
            nc.sum(self._data(a))  # raw numpy arrays are not operands
        with self.assertRaises(ValueError):
            nc.sum(a + nc.rand(3, 4))  # incompatible shapes
        with self.assertRaises(IndexError):
            nc.sum(a[10])  # index out of bounds

all_tests = [NumpyJitTest]
