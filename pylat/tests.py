from unittest import TestCase

from .llvm import Module
from .compile import FunctionCompiler, JitCompiler
from .expr import AssignExpr, ComplexType, Int, IntegerType, Plus, Rational, RealType, Symbol, Times, symbol, S

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

        compiler = JitCompiler()
        builder = FunctionCompiler(compiler)
        fn = builder.compile_assignments([
            AssignExpr(phi, mom_phi * dt)
        ])
        mod = Module()
        mod.add_recursively(values=[fn.fn])

        print()
        for line in fn.print():
            print(line)
        for line in mod.write():
            print(line)

all_tests = [TestExpr, JitTest]
