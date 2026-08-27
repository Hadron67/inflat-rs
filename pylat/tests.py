from typing import Any
from unittest import TestCase

import numpy as np
from llvmlite import binding as llvm
from numpy.testing import assert_almost_equal

from pylat.jit.argpass import ComplexFloatType, FloatType, TypeContext
from pylat.jit.fn_wrapper import Wrapper
from pylat.jit.openmp import OpenMPBackend

from .expr import AssignExpr, Int, Plus, Rational, S, Times, symbols
from .jit.compile import JitCompiler

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
        fn = compiler.compile_assignments([
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context)

        np.random.seed(114514)
        phi0 = np.zeros((10, 10, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(10, 10, 10) + np.random.randn(10, 10, 10) * 1j
        dt0 = 2.0

        fn.call({phi: phi0, mom_phi: mom_phi0, dt: dt0})  # type: ignore

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
        fn = compiler.compile_assignments([
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context)

        np.random.seed(114514)
        phi0 = np.zeros((8, 9, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(8, 9, 10) + np.random.randn(8, 9, 10) * 1j
        dt0 = 2.0

        fn.call({phi: phi0, mom_phi: mom_phi0, dt: dt0})  # type: ignore

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

    def test_sum(self):
        a, = symbols('a')
        context = TypeContext()
        context.set_symbol(a, FloatType(64), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction(a, context)

        np.random.seed(114514)
        a0 = np.random.randn(10, 10, 10)
        result = fn.call({a: a0})  # type: ignore
        assert_almost_equal(result, np.sum(a0))

    def test_sum_complex(self):
        a, b = symbols('a', 'b')
        context = TypeContext()
        context.set_symbol(a, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(b, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction(a * b + a, context)

        np.random.seed(42)
        a0 = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        b0 = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        result = fn.call({a: a0, b: b0})  # type: ignore
        assert_almost_equal(result, np.sum(a0 * b0 + a0))

    def test_sum_with_assignment(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')
        context = TypeContext()
        context.set_symbol(dt, FloatType(64), 0)
        context.set_symbol(phi, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(mom_phi, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context, reduction=mom_phi)

        np.random.seed(114514)
        phi0 = np.zeros((8, 9, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(8, 9, 10) + np.random.randn(8, 9, 10) * 1j
        dt0 = 2.0

        result = fn.call({phi: phi0, mom_phi: mom_phi0, dt: dt0})  # type: ignore

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)
        assert_almost_equal(result, np.sum(mom_phi0))

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

all_tests = [TestExpr, JitTest, JitWrapperTest]
