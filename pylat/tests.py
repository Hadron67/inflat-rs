from unittest import TestCase

import numpy as np
from llvmlite import binding as llvm
from numpy.testing import assert_almost_equal

from pylat.jit.argpass import ComplexFloatType, FloatType, TypeContext
from pylat.jit.openmp import OpenMPBackend

from .expr import AssignExpr, Int, Plus, Rational, S, Times, symbols
from .jit.compile import JitCompiler

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

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

all_tests = [TestExpr, JitTest]
