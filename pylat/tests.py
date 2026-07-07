from unittest import TestCase

from numpy.testing import assert_almost_equal

from pylat.jit.openmp import OpenMPBackend
from pylat.jit.argpass import ComplexFloatType, IntType, TypeContext, FloatType

from .jit.compile import JitCompiler
from .expr import AssignExpr, Int, Plus, Rational, Symbol, Times, symbol, S, symbols

from llvmlite import binding as llvm
import numpy as np

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class TestExpr(TestCase):
    x: Symbol
    y: Symbol
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.x = symbol('x')
        self.y = symbol('y')

    def test_evaluation(self):
        self.assertEqual(
            (self.x + self.x * 2 + S(2) * self.y + self.y * 3 + self.y / 2).evaluate(),
            Plus((Times((Int(3), self.x)), Times((Rational(11, 2), self.y))))
        )

class JitTest(TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_jit(self):
        nx, ny, nz, dt = symbols('nx', 'ny', 'nz', 'dt')
        phi, mom_phi = symbols('phi', 'mom_phi')
        context = TypeContext()
        context.set_symbol(nx, IntType(64, False), ())
        context.set_symbol(ny, IntType(64, False), ())
        context.set_symbol(nz, IntType(64, False), ())
        context.set_symbol(dt, FloatType(64), ())
        context.set_symbol(phi, ComplexFloatType(FloatType(64)), (nz, ny, nx))
        context.set_symbol(mom_phi, ComplexFloatType(FloatType(64)), (nz, ny, nx))

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_one_kernel([
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], context)

        np.random.seed(114514)
        phi0 = np.zeros((10, 10, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(10, 10, 10) + np.random.randn(10, 10, 10) * 1j
        dt0 = 2.0

        fn.call({nz: phi0.shape[2], ny: phi0.shape[1], nx: phi0.shape[0], phi: phi0, mom_phi: mom_phi0, dt: dt0})  # type: ignore

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

all_tests = [TestExpr, JitTest]
