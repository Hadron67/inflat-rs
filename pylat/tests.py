import ctypes
from unittest import TestCase

from pylat.jit.openmp import OpenMPBackend
from pylat.jit.argpass import ComplexFloatType, IntType, TypeContext, FloatType
import pylat.jit.argpass as ap
from pylat.util import add_line_numbers

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
        context.set_symbol(nx, (ap.IntegerType(), IntType(64, False)), ())
        context.set_symbol(ny, (ap.IntegerType(), IntType(64, False)), ())
        context.set_symbol(nz, (ap.IntegerType(), IntType(64, False)), ())
        context.set_symbol(dt, (ap.RealType(), FloatType(64)), ())
        context.set_symbol(phi, (ap.ComplexType(), ComplexFloatType(FloatType(64))), (nz, ny, nx))
        context.set_symbol(mom_phi, (ap.ComplexType(), ComplexFloatType(FloatType(64))), (nz, ny, nx))

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_one_kernel([
            AssignExpr(phi, mom_phi * dt)
        ], context)

        lines = fn.print_all()

        print()
        for line in add_line_numbers(lines):
            print(line)

        phi0 = np.zeros((10, 10, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(10, 10, 10) + np.random.randn(10, 10, 10) * 1j

        fn.call({nz: phi0.shape[2], ny: phi0.shape[1], nx: phi0.shape[0], phi: phi0, mom_phi: mom_phi0, dt: 2})  # type: ignore

all_tests = [TestExpr, JitTest]
