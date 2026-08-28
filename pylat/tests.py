from typing import Any
from unittest import TestCase

import numpy as np
from llvmlite import binding as llvm
from numpy.testing import assert_almost_equal

from pylat.jit.argpass import ComplexFloatType, FloatType, TypeContext
from pylat.jit.fn_wrapper import Wrapper
from pylat.jit.openmp import OpenMPBackend

from .expr import AssignExpr, Int, Plus, Rational, S, Slice, Times, symbols
from .jit.compile import CompiledWrapper, JitCompiler, StandardLayoutMode

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

class TestExpr(TestCase):
    def __init__(self, methodName: str = "test_evaluation") -> None:
        super().__init__(methodName)

    def test_evaluation(self):
        x, y = symbols('x', 'y')
        self.assertEqual(
            (x + x * 2 + S(2) * y + y * 3 + y / 2).normalize(),
            Plus((Times((Int(3), x)), Times((Rational(11, 2), y))))
        )

class JitTest(TestCase):
    def __init__(self, methodName: str = "test_jit") -> None:
        super().__init__(methodName)

    def test_assignment(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')
        context = TypeContext()
        context.set_symbol(dt, FloatType(64), 0)
        context.set_symbol(phi, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(mom_phi, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([phi, mom_phi, dt], [
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context)

        np.random.seed(114514)
        phi0 = np.zeros((10, 10, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(10, 10, 10) + np.random.randn(10, 10, 10) * 1j
        dt0 = 2.0

        fn.call(phi0, mom_phi0, dt0)

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

    def test_assignment_non_uniform_shape(self):
        # regression test: subscripts were unpacked from the shape in the wrong
        # order when the shape was not symmetric under reversal
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')
        context = TypeContext()
        context.set_symbol(dt, FloatType(64), 0)
        context.set_symbol(phi, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(mom_phi, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([phi, mom_phi, dt], [
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context)

        np.random.seed(114514)
        phi0 = np.zeros((8, 9, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(8, 9, 10) + np.random.randn(8, 9, 10) * 1j
        dt0 = 2.0

        fn.call(phi0, mom_phi0, dt0)

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

    def test_sum(self):
        a, = symbols('a')
        context = TypeContext()
        context.set_symbol(a, FloatType(64), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([a], a, context)

        np.random.seed(114514)
        a0 = np.random.randn(10, 10, 10)
        result = fn.call(a0)
        assert_almost_equal(result, np.sum(a0))

    def test_sum_complex(self):
        a, b = symbols('a', 'b')
        context = TypeContext()
        context.set_symbol(a, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(b, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([a, b], a * b + a, context)

        np.random.seed(42)
        a0 = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        b0 = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        result = fn.call(a0, b0)
        assert_almost_equal(result, np.sum(a0 * b0 + a0))

    def test_sum_with_assignment(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')
        context = TypeContext()
        context.set_symbol(dt, FloatType(64), 0)
        context.set_symbol(phi, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(mom_phi, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([phi, mom_phi, dt], [
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context, reduction=mom_phi)

        np.random.seed(114514)
        phi0 = np.zeros((8, 9, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(8, 9, 10) + np.random.randn(8, 9, 10) * 1j
        dt0 = 2.0

        result = fn.call(phi0, mom_phi0, dt0)

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)
        assert_almost_equal(result, np.sum(mom_phi0))

    def test_multiple_assignments_same_shape(self):
        # regression: several assignments accumulating into the same array must
        # resolve the shared shape constraints transitively instead of failing
        a, b, c, d = symbols('a', 'b', 'c', 'd')
        context = TypeContext()
        for s in (a, b, c, d):
            context.set_symbol(s, FloatType(64), 1)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([a, b, c, d], [
            AssignExpr(a, b, '+'),
            AssignExpr(a, c, '+'),
            AssignExpr(a, d, '+'),
        ], context)

        np.random.seed(40)
        a0 = np.zeros(5)
        b0 = np.random.rand(5)
        c0 = np.random.rand(5)
        d0 = np.random.rand(5)
        fn.call(a0, b0, c0, d0)
        assert_almost_equal(a0, b0 + c0 + d0)

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
            a += b[:, 2]
            a += b[1, 3]

        a = np.zeros((5, 6))
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        expected = (
            np.broadcast_to(b[1], a.shape)
            + np.broadcast_to(b[:, 2:3], a.shape)
            + np.full(a.shape, b[1, 3])
        )
        assert_almost_equal(a, expected)

    def test_slice_negative_index(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[-1, :]
            a += b[:, -1]

        a = np.zeros((5, 6))
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        expected = np.broadcast_to(b[-1], a.shape) + np.broadcast_to(b[:, -1:], a.shape)
        assert_almost_equal(a, expected)

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

class SimdLayoutTests(TestCase):
    """Tests for the SIMD friendly linear-index kernels.

    When the expressions contain no rolls and no interior slices, and every array
    argument is a standard layout (C or F contiguous), the kernel skips the
    unpack/repack of the loop index and accesses arrays directly with the flat
    loop variable.
    """

    @staticmethod
    def _last_compiled(f) -> CompiledWrapper:
        return list(f._cache.values())[-1][0]

    def test_row_major_uses_flat_kernel(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c, dt):
            a += b * c + b * dt

        np.random.seed(3)
        a = np.random.rand(8, 9, 10)
        b = np.random.rand(8, 9, 10)
        c = np.random.rand(8, 9, 10)
        a0, b0, c0 = a.copy(), b.copy(), c.copy()
        f(a, b, c, 0.5)
        assert_almost_equal(a, a0 + b0 * c0 + b0 * 0.5)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)
        # the flat kernel must not unpack the loop index (no unsigned div/rem)
        ir = '\n'.join(self._last_compiled(f).print_all())
        self.assertNotIn('urem', ir)
        self.assertNotIn('udiv', ir)

    def test_column_major_uses_flat_kernel(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += b * c + b

        np.random.seed(4)
        a = np.asfortranarray(np.random.rand(6, 7, 8))
        b = np.asfortranarray(np.random.rand(6, 7, 8))
        c = np.asfortranarray(np.random.rand(6, 7, 8))
        a0, b0, c0 = a.copy(), b.copy(), c.copy()
        f(a, b, c)
        assert_almost_equal(a, a0 + b0 * c0 + b0)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.COLUMN_MAJOR)
        ir = '\n'.join(self._last_compiled(f).print_all())
        self.assertNotIn('urem', ir)
        self.assertNotIn('udiv', ir)

    def test_non_contiguous_arrays_fall_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b * 2

        np.random.seed(5)
        a = np.zeros((4, 5))
        b = np.random.rand(5, 4)
        a0 = a.copy()
        f(a, b.T)  # transposed views are not standard layout
        assert_almost_equal(a, a0 + b.T * 2)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_mixed_layout_falls_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += b * c

        np.random.seed(6)
        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        c = np.asfortranarray(np.random.rand(4, 5))
        a0 = a.copy()
        f(a, b, c)
        assert_almost_equal(a, a0 + b * c)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_axis0_slice_uses_flat_kernel(self):
        # slicing the max-stride axis is linear, including negative indices
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1]
            a += b[0]
            a += b[-1]

        a = np.zeros(6)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, b[1] + b[0] + b[-1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

    def test_middle_slice_falls_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[:, 1]

        a = np.zeros(5)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, b[:, 1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_multi_axis_slices_fall_back_to_generic(self):
        # multi-axis slices are nested slices; the sliced sub-expression is not
        # standard layout, so they are compiled by the generic kernel
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1, 2]

        a = np.zeros(7)
        b = np.arange(3 * 4 * 7).reshape(3, 4, 7).astype(float)
        f(a, b)
        assert_almost_equal(a, b[1, 2])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

        @wrapper.jit()
        def g(a, b):
            a += b[1, :, 2]

        a = np.zeros(4)
        b = np.arange(3 * 4 * 7).reshape(3, 4, 7).astype(float)
        g(a, b)
        assert_almost_equal(a, b[1, :, 2])
        self.assertIs(self._last_compiled(g).standard_layout, StandardLayoutMode.NONE)

    def test_roll_falls_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.roll(b, 1, axis=0)

        a = np.zeros((5, 6))
        b = np.random.rand(5, 6)
        f(a, b)
        assert_almost_equal(a, np.roll(b, 1, axis=0))
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_1d_lhs_slice_uses_flat_kernel(self):
        # the sliced base has a higher rank than the loop; the flat kernel handles it
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1] + b[-1]

        a = np.zeros(6)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, b[1] + b[-1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

    def test_slice_of_compound_expression_uses_flat_kernel(self):
        # slicing a function application (or any compound expression over a single
        # array) is linear: np.sin(b)[2][i] == np.sin(b[2][i])
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sin(b)[2]
            a += np.cos(b)[-1]

        np.random.seed(11)
        a = np.zeros(6)
        b = np.random.rand(5, 6) + 1
        f(a, b)
        assert_almost_equal(a, np.sin(b)[2] + np.cos(b)[-1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

    def test_slice_of_compound_expression_no_unpack(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sin(b)[2]

        np.random.seed(12)
        a = np.zeros(6)
        b = np.random.rand(5, 6) + 1
        f(a, b)
        assert_almost_equal(a, np.sin(b)[2])
        ir = '\n'.join(self._last_compiled(f).print_all())
        self.assertNotIn('urem', ir)

    def test_compound_expression_slice_uses_flat_kernel(self):
        # a slice over an expression with several distinct arrays is linear too: each
        # base array gets the fixed index applied with its own stride
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += (b + c)[1]

        np.random.seed(13)
        a = np.zeros(6)
        b = np.random.rand(5, 6)
        c = np.random.rand(5, 6)
        f(a, b, c)
        assert_almost_equal(a, (b + c)[1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

        # the same expression with non-contiguous arrays must stay correct via the
        # generic kernel
        @wrapper.jit()
        def g(a, b):
            a += np.sin(b)[2]

        a = np.zeros(6)
        b = np.random.rand(6, 5)
        g(a, b.T)
        assert_almost_equal(a, np.sin(b.T)[2])
        self.assertIs(self._last_compiled(g).standard_layout, StandardLayoutMode.NONE)

    def test_layout_cache_variants(self):
        # C-contiguous and F-contiguous calls compile separate kernels
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b * 2

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        f(a, b)
        self.assertEqual(len(f._cache), 1)
        a = np.asfortranarray(np.zeros((4, 5)))
        b = np.asfortranarray(np.random.rand(4, 5))
        f(a, b)
        self.assertEqual(len(f._cache), 2)
        self.assertEqual(
            {w.standard_layout for w, _ in f._cache.values()},
            {StandardLayoutMode.ROW_MAJOR, StandardLayoutMode.COLUMN_MAJOR},
        )

    def test_flat_reduction(self):
        a, = symbols('a')
        context = TypeContext()
        context.set_symbol(a, FloatType(64), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([a], a, context, standard_layout=StandardLayoutMode.ROW_MAJOR)
        self.assertIs(fn.standard_layout, StandardLayoutMode.ROW_MAJOR)

        np.random.seed(7)
        a0 = np.random.rand(10, 10, 10)
        assert_almost_equal(fn.call(a0), np.sum(a0))

    def test_flat_reduction_column_major(self):
        a, = symbols('a')
        context = TypeContext()
        context.set_symbol(a, FloatType(64), 2)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([a], a, context, standard_layout=StandardLayoutMode.COLUMN_MAJOR)
        self.assertIs(fn.standard_layout, StandardLayoutMode.COLUMN_MAJOR)

        np.random.seed(8)
        a0 = np.asfortranarray(np.random.rand(10, 10))
        assert_almost_equal(fn.call(a0), np.sum(a0))

    def test_flat_reduction_with_slice(self):
        b, = symbols('b')
        context = TypeContext()
        context.set_symbol(b, FloatType(64), 2)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([b], Slice(b, 0, 1), context, standard_layout=StandardLayoutMode.ROW_MAJOR)
        self.assertIs(fn.standard_layout, StandardLayoutMode.ROW_MAJOR)

        np.random.seed(9)
        b0 = np.random.rand(5, 6)
        assert_almost_equal(fn.call(b0), np.sum(b0[1]))

    def test_mismatched_layout_raises(self):
        # the direct JitCompiler API verifies the layout of the arguments at call time
        a, b = symbols('a', 'b')
        context = TypeContext()
        context.set_symbol(a, FloatType(64), 2)
        context.set_symbol(b, FloatType(64), 2)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments(
            [a, b], [AssignExpr(a, b, '+')], context, standard_layout=StandardLayoutMode.ROW_MAJOR
        )
        a0 = np.zeros((4, 5))
        b0 = np.asfortranarray(np.random.rand(4, 5))
        with self.assertRaises(ValueError):
            fn.call(a0, b0)

    def test_generic_rank_mismatched_slices(self):
        # regression: the generic kernel must handle sliced arrays whose rank is
        # higher than the loop rank (e.g. a += b[1] with a one-dimensional a)
        a, b = symbols('a', 'b')
        context = TypeContext()
        context.set_symbol(a, FloatType(64), 1)
        context.set_symbol(b, FloatType(64), 2)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([a, b], [AssignExpr(a, Slice(b, 0, 1), '+')], context)
        a0 = np.zeros(6)
        b0 = np.arange(30).reshape(5, 6).astype(float)
        fn.call(a0, b0)
        assert_almost_equal(a0, b0[1])

        context3 = TypeContext()
        context3.set_symbol(a, FloatType(64), 1)
        b3, = symbols('b3')
        context3.set_symbol(b3, FloatType(64), 3)
        fn3 = compiler.compile_assignments(
            [a, b3], [AssignExpr(a, Slice(Slice(b3, 2, 2), 0, 1), '+')], context3
        )
        a3 = np.zeros(4)
        b30 = np.arange(3 * 4 * 7).reshape(3, 4, 7).astype(float)
        fn3.call(a3, b30)
        assert_almost_equal(a3, b30[1, :, 2])

        # a sliced array that is broadcast into a higher-rank loop (base rank is
        # smaller than the loop rank) must align its surviving axis with the
        # trailing loop axes
        context4 = TypeContext()
        context4.set_symbol(a, FloatType(64), 3)
        context4.set_symbol(b, FloatType(64), 2)
        fn4 = compiler.compile_assignments([a, b], [AssignExpr(a, Slice(b, 0, 1), '+')], context4)
        a4 = np.zeros((2, 3, 6))
        b40 = np.arange(30).reshape(5, 6).astype(float)
        fn4.call(a4, b40)
        assert_almost_equal(a4, np.broadcast_to(b40[1], a4.shape))

        context5 = TypeContext()
        context5.set_symbol(a, FloatType(64), 2)
        b5, = symbols('b5')
        context5.set_symbol(b5, FloatType(64), 4)
        fn5 = compiler.compile_assignments(
            [a, b5], [AssignExpr(a, Slice(Slice(b5, 2, 3), 1, 2), '+')], context5
        )
        a5 = np.zeros((2, 4))
        b50 = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(float)
        fn5.call(a5, b50)
        assert_almost_equal(a5, b50[:, 2, 3][:, :4])

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

all_tests = [TestExpr, JitTest, JitWrapperTest, SimdLayoutTests, ObjectInliningTest]
