from unittest import TestCase

from numpy.testing import assert_almost_equal

from pylat.jit.openmp import OpenMPBackend
from pylat.jit.argpass import ComplexFloatType, TypeContext, FloatType

from .jit.compile import JitCompiler
from .expr import AssignExpr, Int, Plus, Rational, Times, S, symbols

from llvmlite import binding as llvm
import numpy as np

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class TestExpr(TestCase):
    def __init__(self, methodName: str = "test_evaluation") -> None:
        super().__init__(methodName)

    def test_evaluation(self):
        x, y = symbols('x', 'y')
        self.assertEqual(
            (x + x * 2 + S(2) * y + y * 3 + y / 2).evaluate(),
            Plus((Times((Int(3), x)), Times((Rational(11, 2), y))))
        )

class JitTest(TestCase):
    def __init__(self, methodName: str = "test_jit") -> None:
        super().__init__(methodName)

    def test_jit(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')
        context = TypeContext()
        context.set_symbol(dt, FloatType(64), 0)
        context.set_symbol(phi, ComplexFloatType(FloatType(64)), 3)
        context.set_symbol(mom_phi, ComplexFloatType(FloatType(64)), 3)

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_one_kernel([
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context)

        np.random.seed(114514)
        phi0 = np.zeros((10, 10, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(10, 10, 10) + np.random.randn(10, 10, 10) * 1j
        dt0 = 2.0

        fn.call({phi: phi0, mom_phi: mom_phi0, dt: dt0})  # type: ignore

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

all_tests = [TestExpr, JitTest]
