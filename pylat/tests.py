from unittest import TestCase

from pylat.jit.openmp import OpenMPBackend
from pylat.util import add_line_numbers

from .jit.llvm import Module
from .jit.compile import JitCompiler
from .expr import AssignExpr, ComplexType, Int, IntegerType, Plus, Rational, RealType, Symbol, Times, symbol, S

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
        nx = Symbol(('nx',), type=IntegerType(), shape=())
        ny = Symbol(('ny',), type=IntegerType(), shape=())
        nz = Symbol(('nz',), type=IntegerType(), shape=())
        dt = Symbol(('dt',), shape=())
        phi = Symbol(('phi',), type=ComplexType(), shape=(nz, ny, nx))
        mom_phi = Symbol(('mom_phi',), type=ComplexType(), shape=(nz, ny, nx))

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_one_kernel([
            AssignExpr(phi, mom_phi * dt)
        ])

        lines = fn.print_all()

        print()
        for line in add_line_numbers(lines):
            print(line)

        mod = llvm.parse_assembly('\n'.join(lines))
        mod.verify()


all_tests = [TestExpr, JitTest]
