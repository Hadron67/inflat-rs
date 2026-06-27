from .expr import Int, Plus, Rational, Symbol, Times, symbol
from unittest import TestCase

class TestExpr(TestCase):
    x: Symbol
    y: Symbol
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.x = symbol('x')
        self.y = symbol('y')

    def test_evaluation(self):
        self.assertEqual(
            (self.x + self.x + self.y + self.y + self.y / 2).evaluate(),
            Plus((Times((Int(2), self.x)), Times((Rational(5, 2), self.y))))
        )

all_tests = [TestExpr]
