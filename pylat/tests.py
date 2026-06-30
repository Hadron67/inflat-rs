from unittest import TestCase

from pylat.jit import typed
from pylat.jit.openmp import OpenMPBackend
from pylat.jit.typed import ComplexFloatType, IntType, TypeContext, FloatType
from pylat.util import add_line_numbers

from .jit.compile import JitCompiler
from .expr import AssignExpr, Int, Plus, Rational, Symbol, Times, symbol, S, symbols

from llvmlite import binding as llvm

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
        # phi = Symbol(('phi',), type=ComplexType(), shape=(nz, ny, nx))
        # mom_phi = Symbol(('mom_phi',), type=ComplexType(), shape=(nz, ny, nx))
        context = TypeContext()
        context.set_symbol(nx, (typed.IntegerType(), IntType(64, False)), ())
        context.set_symbol(ny, (typed.IntegerType(), IntType(64, False)), ())
        context.set_symbol(nz, (typed.IntegerType(), IntType(64, False)), ())
        context.set_symbol(dt, (typed.RealType(), FloatType(64)), ())
        context.set_symbol(phi, (typed.ComplexType(), ComplexFloatType(64)), (nz, ny, nx))
        context.set_symbol(mom_phi, (typed.ComplexType(), ComplexFloatType(64)), (nz, ny, nx))

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_one_kernel([
            AssignExpr(phi, mom_phi * dt)
        ], context)

        lines = fn.print_all()

        print()
        for line in add_line_numbers(lines):
            print(line)

all_tests = [TestExpr, JitTest]
