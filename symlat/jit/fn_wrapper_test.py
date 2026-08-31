import ctypes
from typing import Any
from unittest import TestCase

import numpy as np
from llvmlite import binding as llvm
from numpy.testing import assert_almost_equal

from ..expr import coord, coords
from .fn_wrapper import Wrapper

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


def _jitted_from_source(wrapper: Wrapper, source: str, name: str):
    """Build a jittable function from source text.

    Used for tests whose function bodies are intentionally invalid Python
    (undefined names etc.), so that static analyzers do not flag them.
    """
    namespace: dict[str, Any] = {'__name__': 'jittest'}
    exec(compile(source, '<jittest>', 'exec'), namespace)  # noqa: S102
    return wrapper.jit()(namespace[name])

class JitWrapperTest(TestCase):
    def test_usage_example(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def my_func(a, b, c, dt):
            a += c * dt
            b += c * dt + c * 2

        np.random.seed(114514)
        a = np.random.rand(8, 9, 10)
        b = np.random.rand(8, 9, 10)
        c = np.random.rand(8, 9, 10)
        a0 = a.copy()
        b0 = b.copy()
        dt = 0.5
        my_func(a, b, c, dt)
        assert_almost_equal(a, a0 + c * dt)
        assert_almost_equal(b, b0 + c * dt + c * 2)

    def test_scalar_ref_argument(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(dt, scale):
            dt *= scale

        dt = ctypes.c_double(3.0)
        scale = ctypes.c_double(2.0)
        f(dt, scale)
        self.assertEqual(dt.value, 6.0)
        self.assertEqual(scale.value, 2.0)

    def test_scalar_ref_read_only(self):
        # a reference scalar that is never written to is compiled by value
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, dt):
            a += dt

        a = np.zeros(5)
        dt = ctypes.c_double(3.0)
        f(a, dt)
        assert_almost_equal(a, np.full(5, 3.0))
        self.assertEqual(dt.value, 3.0)

    def test_zero_d_array_argument(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, dt):
            a += dt

        a = np.zeros(5)
        dt = np.array(3.0)
        f(a, dt)
        assert_almost_equal(a, np.full(5, 3.0))

    def test_write_to_numpy_scalar_raises(self):
        # numpy scalars have no writable address: writing to one is a compile error
        wrapper = Wrapper()

        @wrapper.jit()
        def f(dt):
            dt += 1.0

        with self.assertRaises(TypeError):
            f(np.float64(2.0))

    def test_shape_and_dtype_variants(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, dt):
            a += b * dt

        # different shapes reuse the compiled kernel (shape is a runtime argument)
        a = np.random.rand(3, 4)
        b = np.random.rand(3, 4)
        a0 = a.copy()
        f(a, b, 2.0)
        assert_almost_equal(a, a0 + b * 2.0)
        # float32 arrays with a float64 scalar
        a = np.random.rand(2, 3).astype(np.float32)
        b = np.random.rand(2, 3).astype(np.float32)
        a0 = a.copy()
        f(a, b, 1.5)
        assert_almost_equal(a, a0 + b * 1.5)
        # integer scalar parameter
        a = np.random.rand(2, 3)
        b = np.random.rand(2, 3)
        a0 = a.copy()
        f(a, b, 3)
        assert_almost_equal(a, a0 + b * 3)

    def test_complex_arrays(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def step(phi, mom, dt):
            phi += mom * dt

        phi = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        mom = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        phi0 = phi.copy()
        step(phi, mom, 0.3)
        assert_almost_equal(phi, phi0 + mom * 0.3)

    def test_numeric_functions_and_unary_minus(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(x, y):
            x += np.sin(y) + np.cos(y) + np.exp(y) + np.log(y)
            y *= -x

        x = np.random.rand(6, 6) + 1
        y = np.random.rand(6, 6) + 0.5
        x0, y0 = x.copy(), y.copy()
        f(x, y)
        assert_almost_equal(x, x0 + np.sin(y0) + np.cos(y0) + np.exp(y0) + np.log(y0))
        assert_almost_equal(y, y0 * -x)

    def test_numpy_scalar_operands(self):
        # numpy scalars and reversed operands dispatch through __array_ufunc__
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += 2 * b + np.float64(3) * b + np.int64(4) * b + b ** 2 - b / 2

        np.random.seed(0)
        a = np.random.rand(5, 5)
        b = np.random.rand(5, 5)
        a0 = a.copy()
        f(a, b)
        assert_almost_equal(a, a0 + 2 * b + 3 * b + 4 * b + b ** 2 - b / 2)

    def test_roll(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.roll(b, 1, axis=0)
            a += np.roll(b, -1, axis=1)
            a += np.roll(b, 7, axis=1)  # shift larger than the axis length

        a = np.zeros((5, 6))
        b = np.random.rand(5, 6)
        f(a, b)
        assert_almost_equal(a, np.roll(b, 1, axis=0) + np.roll(b, -1, axis=1) + np.roll(b, 7, axis=1))

    def test_roll_multiple_axes(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.roll(b, (1, -2), axis=(0, 1))

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        f(a, b)
        assert_almost_equal(a, np.roll(b, (1, -2), axis=(0, 1)))

    def test_slice(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1]
            a += b[1, 3]

        a = np.zeros((5, 6))
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, np.broadcast_to(b[1], a.shape) + np.full(a.shape, b[1, 3]))

        # a trailing-axis slice has a different shape than a row slice, so it
        # needs its own loop (and its own jitted function)
        @wrapper.jit()
        def g(c, b):
            c += b[:, 2]

        c = np.zeros(5)
        g(c, b)
        assert_almost_equal(c, b[:, 2])

    def test_slice_negative_index(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[-1, :]

        a = np.zeros((5, 6))
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, np.broadcast_to(b[-1], a.shape))

        @wrapper.jit()
        def g(c, b):
            c += b[:, -1]

        c = np.zeros(5)
        g(c, b)
        assert_almost_equal(c, b[:, -1])

    def test_roll_and_slice_combined(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.roll(b, 1, axis=0) + np.roll(b, -1, axis=0) - 2 * b
            a += b[0] - b[-1]

        a = np.zeros((6, 8))
        b = np.random.rand(6, 8)
        f(a, b)
        expected = (
            (np.roll(b, 1, axis=0) + np.roll(b, -1, axis=0) - 2 * b)
            + np.broadcast_to(b[0] - b[-1], a.shape)
        )
        assert_almost_equal(a, expected)

    def test_roll_slice_errors(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f1(a, b):
            a += np.roll(b, 1)  # axis is required

        with self.assertRaises(TypeError):
            f1(np.zeros((3, 3)), np.zeros((3, 3)))

        @wrapper.jit()
        def f2(a, b):
            a += b[1:3]  # range slices are not supported

        with self.assertRaises(TypeError):
            f2(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_flip(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.flip(b, axis=0)
            a += np.flip(b, axis=1)
            a += np.flip(b, axis=-1)  # negative axis

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        f(a, b)
        assert_almost_equal(a, np.flip(b, axis=0) + np.flip(b, axis=1) + np.flip(b, axis=-1))

    def test_flip_multiple_axes(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.flip(b, axis=(0, 1))
            a += np.flip(b)  # axis=None flips every axis

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        f(a, b)
        assert_almost_equal(a, np.flip(b, axis=(0, 1)) + np.flip(b))

    def test_flip_rank_one(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.flip(b, axis=0)

        a = np.zeros(6)
        b = np.arange(6).astype(float)
        f(a, b)
        assert_almost_equal(a, np.flip(b))

    def test_flip_nested(self):
        # nested flips merge during normalization: flipping the same axis twice
        # cancels out
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.flip(np.flip(b, axis=0), axis=0)  # cancels to b
            a += np.flip(np.flip(b, axis=0), axis=1)  # merges to (0, 1)

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        f(a, b)
        assert_almost_equal(a, b + np.flip(b, axis=(0, 1)))

    def test_flip_broadcast(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.flip(b, axis=0)

        a = np.zeros((4, 5))
        b = np.arange(5).astype(float)  # rank-1 argument broadcast along axis 0
        f(a, b)
        assert_almost_equal(a, np.broadcast_to(np.flip(b), a.shape))

    def test_flip_derived_array(self):
        # axis=None (flip every axis) also works on intermediate expressions,
        # whose rank is not known while tracing
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.flip(b[0])
            a += np.flip(b, axis=0)[0]

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        b0 = b.copy()
        f(a, b)
        expected = (
            np.broadcast_to(np.flip(b0[0]), a.shape)
            + np.broadcast_to(np.flip(b0, axis=0)[0], a.shape)
        )
        assert_almost_equal(a, expected)

    def test_flip_in_sum(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sum(b * np.flip(b, axis=1))

        a = np.zeros((3, 4))
        b = np.random.rand(3, 4)
        b0 = b.copy()
        f(a, b)
        assert_almost_equal(a, np.full(a.shape, np.sum(b0 * np.flip(b0, axis=1))))

    def test_flip_errors(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f1(a, b):
            a += np.flip(b, axis=2)  # axis out of bounds

        with self.assertRaises(TypeError):
            f1(np.zeros((3, 3)), np.zeros((3, 3)))

        @wrapper.jit()
        def f2(a, b, c):
            a += np.flip(b, axis=c)  # traced axis is not a compile-time constant

        with self.assertRaises(TypeError):
            f2(np.zeros((3, 3)), np.zeros((3, 3)), 0)

    def test_coord(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            x = coords(a.shape)
            a += b * x[0] + x[1]
            b += x[0] * x[1]

        a = np.zeros((4, 5))
        b = np.ones((4, 5))
        b0 = b.copy()
        f(a, b)
        i = np.arange(4)[:, None]
        j = np.arange(5)[None, :]
        assert_almost_equal(a, b0 * i + j)
        assert_almost_equal(b, b0 + i * j)

    def test_coord_rank_one(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a):
            a += coord(0, a.shape)

        a = np.zeros(6)
        f(a)
        assert_almost_equal(a, np.arange(6))

    def test_coord_in_expression(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            x = coords(a.shape)
            a += b * x[0] + np.roll(b, 1, axis=0) * 0.5
            a += b[0] * x[1]

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        b0 = b.copy()
        f(a, b)
        i = np.arange(4)[:, None]
        j = np.arange(5)[None, :]
        expected = i * b0 + np.roll(b0, 1, axis=0) * 0.5 + np.broadcast_to(b0[0], a.shape) * j
        assert_almost_equal(a, expected)

    def test_coord_left_operand(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            x = coords(a.shape)
            a += x[0] * b + 2 * x[1]

        a = np.zeros((4, 5))
        b = np.ones((4, 5))
        b0 = b.copy()
        f(a, b)
        i = np.arange(4)[:, None]
        j = np.arange(5)[None, :]
        assert_almost_equal(a, i * b0 + 2 * j)

    def test_coord_in_sum(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sum(b * coord(0, b.shape))

        a = np.zeros((3, 4))
        b = np.random.rand(3, 4)
        b0 = b.copy()
        f(a, b)
        i = np.arange(3)[:, None]
        assert_almost_equal(a, np.full(a.shape, np.sum(b0 * i)))

    def test_coord_axis_out_of_bounds(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a):
            a += coord(2, a.shape)

        with self.assertRaises(TypeError):
            f(np.zeros((4, 5)))

    def test_sum(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sum(b)

        np.random.seed(40)
        a = np.zeros(5)
        b = np.random.rand(5)
        f(a, b)
        assert_almost_equal(a, np.full(5, np.sum(b)))
        a = np.zeros((3, 4))
        b = np.random.rand(3, 4)
        f(a, b)
        assert_almost_equal(a, np.full((3, 4), np.sum(b)))

    def test_sum_of_expression(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += np.sum(b * 2) + np.sum(c)

        np.random.seed(41)
        a = np.zeros(6)
        b = np.random.rand(6)
        c = np.random.rand(6)
        f(a, b, c)
        assert_almost_equal(a, np.full(6, np.sum(b * 2) + np.sum(c)))

    def test_sum_with_roll(self):
        # a reduction whose summand needs the generic (non-standard-layout) kernel
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sum(np.roll(b, 1, axis=0))

        np.random.seed(42)
        a = np.zeros((5, 6))
        b = np.random.rand(5, 6)
        f(a, b)
        assert_almost_equal(a, np.full((5, 6), np.sum(np.roll(b, 1, axis=0))))

    def test_sum_int(self):
        # integer sums follow the C convention of being signed
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sum(b)

        a = np.zeros(4)
        b = np.array([1, 2, 3, 4])
        f(a, b)
        assert_almost_equal(a, np.full(4, np.sum(b)))

    def test_sum_errors(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f1(a, b):
            a += np.sum(b, axis=0)  # only the sum over all axes is supported

        with self.assertRaises(TypeError):
            f1(np.zeros((3, 3)), np.zeros((3, 3)))

        @wrapper.jit()
        def f2(a, b):
            a += np.sum(np.sum(b))  # nested sums are not supported

        with self.assertRaises(TypeError):
            f2(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_plain_assignment(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a[:] = b + c

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        c = np.random.rand(4, 5)
        f(a, b, c)
        assert_almost_equal(a, b + c)

    def test_local_variables(self):
        # intermediate locals build expression trees that are used by later
        # in-place updates
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c, dt):
            tmp = c * dt
            a += tmp
            b += tmp * 2

        a = np.random.rand(3, 4)
        b = np.random.rand(3, 4)
        c = np.random.rand(3, 4)
        a0, b0 = a.copy(), b.copy()
        f(a, b, c, 2.0)
        assert_almost_equal(a, a0 + c * 2.0)
        assert_almost_equal(b, b0 + c * 2.0 * 2)

    def test_closures_are_supported(self):
        # constants captured from the enclosing scope are baked into the kernel
        wrapper = Wrapper()
        coeff = 1.5

        @wrapper.jit()
        def f(a, b):
            a += b * coeff

        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        a0 = a.copy()
        f(a, b)
        assert_almost_equal(a, a0 + b * coeff)

    def test_sub_function_calls(self):
        # plain helper functions receive probes and return probe expressions
        wrapper = Wrapper()

        def bar(x):
            return x * 2

        @wrapper.jit()
        def foo(a, b):
            a += bar(b)

        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        a0 = a.copy()
        foo(a, b)
        assert_almost_equal(a, a0 + b * 2)

    def test_nested_sub_function_calls(self):
        # sub-functions may call further sub-functions and take several arguments
        wrapper = Wrapper()

        def baz(x, y):
            return x * y

        def bar(x, k):
            return baz(x, x) + k

        @wrapper.jit()
        def foo(a, b):
            a += bar(b, 1.5)

        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        a0 = a.copy()
        foo(a, b)
        assert_almost_equal(a, a0 + b * b + 1.5)

    def test_sub_function_with_numpy_functions(self):
        wrapper = Wrapper()

        def bar(x):
            return np.sin(x) + np.exp(x)

        @wrapper.jit()
        def foo(a, b):
            a += bar(b)

        a = np.random.rand(4, 4) + 1
        b = np.random.rand(4, 4) + 0.5
        a0 = a.copy()
        foo(a, b)
        assert_almost_equal(a, a0 + np.sin(b) + np.exp(b))

    def test_sub_function_lambda(self):
        wrapper = Wrapper()
        triple = lambda x: x * 3

        @wrapper.jit()
        def foo(a, b):
            a += triple(b)

        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        a0 = a.copy()
        foo(a, b)
        assert_almost_equal(a, a0 + b * 3)

    def test_sub_function_returning_constant(self):
        wrapper = Wrapper()

        def bar():
            return 2.5

        @wrapper.jit()
        def foo(a, b):
            a += b * bar()

        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        a0 = a.copy()
        foo(a, b)
        assert_almost_equal(a, a0 + b * 2.5)

    def test_unused_parameters(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += 1

        a = np.zeros((3, 3))
        f(a, np.random.rand(3, 3))
        assert_almost_equal(a, np.ones((3, 3)))

    def test_compile_cache(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, dt):
            a += b * dt

        a = np.random.rand(4, 4)
        b = np.random.rand(4, 4)
        f(a, b, 1.0)
        f(a, b, 1.0)
        # same (dtype, rank) signature is compiled only once
        self.assertEqual(len(f._cache), 1)
        a = np.random.rand(4, 4).astype(np.float32)
        b = np.random.rand(4, 4).astype(np.float32)
        f(a, b, 1.0)
        self.assertEqual(len(f._cache), 2)

    def test_comptime_args_by_name(self):
        wrapper = Wrapper()

        @wrapper.jit(comptime_args={'dt'})
        def f(a, b, dt):
            a += b * dt

        np.random.seed(20)
        b = np.random.rand(5, 6)
        a = np.zeros((5, 6))
        f(a, b, 2.0)
        assert_almost_equal(a, b * 2.0)
        # a different compile-time value compiles a separate kernel
        a = np.zeros((5, 6))
        f(a, b, 3.0)
        assert_almost_equal(a, b * 3.0)
        # the same value reuses the cached kernel
        a = np.zeros((5, 6))
        f(a, b, 2.0)
        assert_almost_equal(a, b * 2.0)
        self.assertEqual(len(f._cache), 2)

    def test_comptime_args_by_position(self):
        wrapper = Wrapper()

        @wrapper.jit(comptime_args={1})
        def f(x, n):
            x *= n

        x = np.ones((4, 4))
        f(x, 3)
        assert_almost_equal(x, np.full((4, 4), 3))
        x = np.ones((4, 4))
        f(x, 5)
        assert_almost_equal(x, np.full((4, 4), 5))

    def test_comptime_slice_index(self):
        # slice indices that used to require literals can be comptime parameters
        wrapper = Wrapper()

        @wrapper.jit(comptime_args={'idx'})
        def f(a, b, idx):
            a += b[idx]

        a = np.zeros(6)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b, 1)
        assert_almost_equal(a, b[1])
        a = np.zeros(6)
        f(a, b, -1)
        assert_almost_equal(a, b[-1])
        self.assertEqual(len(f._cache), 2)

    def test_comptime_control_flow(self):
        # comptime values enable compile-time conditionals: only the taken branch
        # is traced and compiled
        wrapper = Wrapper()

        @wrapper.jit(comptime_args={'mode'})
        def f(a, b, c, mode):
            if mode == 1:
                a += b
            else:
                a += c

        a = np.zeros(5)
        b = np.random.rand(5)
        c = np.random.rand(5)
        f(a, b, c, 1)
        assert_almost_equal(a, b)
        a = np.zeros(5)
        f(a, b, c, 2)
        assert_almost_equal(a, c)
        self.assertEqual(len(f._cache), 2)

    def test_comptime_unsupported_hashable(self):
        # arguments that cannot be passed as runtime scalars but are hashable are
        # baked as compile-time constants (e.g. tuples for np.roll)
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, shift):
            a += np.roll(b, shift, axis=0)

        a = np.zeros((5, 6))
        b = np.random.rand(5, 6)
        f(a, b, (1,))
        assert_almost_equal(a, np.roll(b, 1, axis=0))
        a = np.zeros((5, 6))
        f(a, b, (2,))
        assert_almost_equal(a, np.roll(b, 2, axis=0))
        self.assertEqual(len(f._cache), 2)

    def test_comptime_unhashable_error(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, bad):
            a += 1

        with self.assertRaises(TypeError):
            f(np.zeros((3, 3)), [1, 2])

    def test_varargs(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, *rest):
            for x in rest:
                a += x

        np.random.seed(30)
        a = np.zeros(5)
        b = np.random.rand(5)
        f(a, b)
        assert_almost_equal(a, b)
        a = np.zeros(5)
        c = np.random.rand(5)
        f(a, b, c)
        assert_almost_equal(a, b + c)
        a = np.zeros(5)
        f(a, b, c, b)
        assert_almost_equal(a, b + c + b)
        # different numbers of variadic arguments compile separate kernels
        self.assertEqual(len(f._cache), 3)

    def test_varargs_indexing(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, *rest):
            a += rest[0]
            a += rest[1] * 2

        np.random.seed(31)
        a = np.zeros(5)
        b = np.random.rand(5)
        c = np.random.rand(5)
        f(a, b, c)
        assert_almost_equal(a, b + c * 2)

    def test_kwargs(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, **kw):
            a += kw['x'] * kw['y']

        np.random.seed(32)
        a = np.zeros(5)
        b = np.random.rand(5)
        c = np.random.rand(5)
        f(a, x=b, y=c)
        assert_almost_equal(a, b * c)

    def test_kwargs_variants_compile_separately(self):
        # different keyword names are part of the JIT cache key
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, **kw):
            for x in kw.values():
                a += x

        np.random.seed(33)
        a = np.zeros(5)
        b = np.random.rand(5)
        f(a, u=b)
        assert_almost_equal(a, b)
        a = np.zeros(5)
        c = np.random.rand(5)
        f(a, v=c)
        assert_almost_equal(a, c)
        self.assertEqual(len(f._cache), 2)

    def test_varargs_and_kwargs(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, *rest, **kw):
            for x in rest:
                a += x
            a += kw['k']

        np.random.seed(34)
        a = np.zeros(5)
        b = np.random.rand(5)
        c = np.random.rand(5)
        d = np.random.rand(5)
        f(a, b, c, k=d)
        assert_almost_equal(a, b + c + d)
        a = np.zeros(5)
        f(a, b, k=d)
        assert_almost_equal(a, b + d)
        self.assertEqual(len(f._cache), 2)

    def test_unexpected_kwargs_rejected(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a):
            a += 1

        with self.assertRaises(TypeError):
            f(np.zeros((3, 3)), dt=1.0)

    def test_varargs_comptime_fallback(self):
        # hashable arguments that cannot be passed at runtime are compile-time,
        # even when they arrive through *varargs
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, *rest):
            a += np.roll(b, rest[0], axis=0)

        a = np.zeros((5, 6))
        b = np.random.rand(5, 6)
        f(a, b, (1,))
        assert_almost_equal(a, np.roll(b, 1, axis=0))
        a = np.zeros((5, 6))
        f(a, b, (2,))
        assert_almost_equal(a, np.roll(b, 2, axis=0))
        self.assertEqual(len(f._cache), 2)

    def test_trace_errors(self):
        wrapper = Wrapper()
        # tracing happens lazily on the first call; errors surface there
        # undefined names raise while the function is traced (it is actually called)
        f1 = _jitted_from_source(wrapper, 'def f1(a, b):\n    a += undefined_name\n', 'f1')
        with self.assertRaises(NameError):
            f1(np.zeros((3, 3)), np.zeros((3, 3)))
        # name that is not a parameter
        f2 = _jitted_from_source(wrapper, 'def f2(a):\n    a += b\n', 'f2')
        with self.assertRaises(NameError):
            f2(np.zeros((3, 3)))
        # returning a value is not allowed; mutate in place instead
        f3 = _jitted_from_source(wrapper, 'def f3(a, b):\n    return a + b\n', 'f3')
        with self.assertRaises(TypeError):
            f3(np.zeros((3, 3)), np.zeros((3, 3)))
        # comparisons are not supported
        f4 = _jitted_from_source(wrapper, 'def f4(a):\n    if a > 0:\n        pass\n', 'f4')
        with self.assertRaises(TypeError):
            f4(np.zeros((3, 3)))
        # no assignments in the body
        f5 = _jitted_from_source(wrapper, 'def f5(a):\n    pass\n', 'f5')
        with self.assertRaises(TypeError):
            f5(np.zeros((3, 3)))
        # updating an intermediate expression is not allowed
        f6 = _jitted_from_source(wrapper, 'def f6(a, b):\n    tmp = a + b\n    tmp += a\n', 'f6')
        with self.assertRaises(TypeError):
            f6(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_jitted_function_cannot_be_nested(self):
        # helper sub-functions must be plain functions, not jitted ones
        wrapper = Wrapper()

        @wrapper.jit()
        def bar(x):
            x += 1

        @wrapper.jit()
        def foo(a):
            bar(a)

        with self.assertRaises(TypeError):
            foo(np.zeros((3, 3)))
class ObjectInliningTest(TestCase):
    def __init__(self, methodName: str = "test_object_inlining") -> None:
        super().__init__(methodName)

    def test_object_inlining(self):
        wrapper = Wrapper()

        class Test:
            def __init__(self, a: np.ndarray, b: np.ndarray):
                self.a = a
                self.b = b

            @wrapper.jit()
            def run(self, dt: float):
                self.a += self.b * dt

        np.random.seed(0)
        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        Test(a, b).run(2.0)
        assert_almost_equal(a, b * 2.0)

    def test_object_inlining_with_varargs(self):
        wrapper = Wrapper()

        class Test:
            def __init__(self, a: np.ndarray, b: np.ndarray):
                self.a = a
                self.b = b

            @wrapper.jit()
            def run(self, *rest):
                self.a += self.b * rest[0] + rest[1]

        np.random.seed(1)
        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        c = np.random.rand(4, 5)
        Test(a, b).run(2.0, c)
        assert_almost_equal(a, b * 2.0 + c)

    def test_object_inlining_with_kwargs(self):
        wrapper = Wrapper()

        class Test:
            def __init__(self, a: np.ndarray, b: np.ndarray):
                self.a = a
                self.b = b

            @wrapper.jit()
            def run(self, **kw):
                self.a += self.b * kw['dt']

        np.random.seed(2)
        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        Test(a, b).run(dt=2.0)
        assert_almost_equal(a, b * 2.0)

    def test_object_inlining_with_varargs_and_kwargs(self):
        wrapper = Wrapper()

        class Test:
            def __init__(self, a: np.ndarray, b: np.ndarray):
                self.a = a
                self.b = b

            @wrapper.jit()
            def run(self, *rest, **kw):
                self.a += self.b * rest[0] + kw['c']

        np.random.seed(3)
        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        c = np.random.rand(4, 5)
        Test(a, b).run(2.0, c=c)
        assert_almost_equal(a, b * 2.0 + c)
        self.assertEqual(len(Test.run._cache), 1)

    def test_object_inlining_in_varargs(self):
        # an object arriving through *varargs is inlined as well
        wrapper = Wrapper()

        class Test:
            def __init__(self, a: np.ndarray, b: np.ndarray):
                self.a = a
                self.b = b

        @wrapper.jit()
        def f(x, *rest):
            x += rest[0].a * rest[0].b

        np.random.seed(4)
        x = np.zeros(6)
        a = np.random.rand(6)
        b = np.random.rand(6)
        f(x, Test(a, b))
        assert_almost_equal(x, a * b)

    def test_object_inlining_in_kwargs(self):
        # an object arriving as a **kwargs value is inlined as well
        wrapper = Wrapper()

        class Test:
            def __init__(self, a: np.ndarray, b: np.ndarray):
                self.a = a
                self.b = b

        @wrapper.jit()
        def f(x, **kw):
            x += kw['obj'].a * kw['obj'].b

        np.random.seed(5)
        x = np.zeros(6)
        a = np.random.rand(6)
        b = np.random.rand(6)
        f(x, obj=Test(a, b))
        assert_almost_equal(x, a * b)

all_tests = [JitWrapperTest, ObjectInliningTest]
