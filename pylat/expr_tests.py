from unittest import TestCase

from pylat.expr import (
    Complex,
    Context,
    Cos,
    Exp,
    Float,
    Int,
    Ln,
    Plus,
    Power,
    Rational,
    Roll,
    S,
    Sin,
    Slice,
    Symbol,
    SymbolShape,
    Times,
    symbols,
)

x, y, z = symbols('x', 'y', 'z')

class ConstantEvaluationTests(TestCase):
    def test_int(self):
        self.assertEqual(Int(3).evaluate(), Int(3))
        self.assertEqual(Int(0).evaluate(), Int(0))
        self.assertEqual(Int(-7).evaluate(), Int(-7))

    def test_float(self):
        # integral floats are normalized to Int
        self.assertEqual(S(2.0).evaluate(), Int(2))
        self.assertEqual(S(0.0).evaluate(), Int(0))
        self.assertEqual(S(-3.0).evaluate(), Int(-3))
        # non-integral floats are kept
        self.assertEqual(S(2.5).evaluate(), Float(2.5))
        self.assertEqual(S(-0.5).evaluate(), Float(-0.5))

    def test_rational(self):
        # fraction reduction
        self.assertEqual(Rational(6, 4).evaluate(), Rational(3, 2))
        self.assertEqual(Rational(2, 4).evaluate(), Rational(1, 2))
        # zero numerator
        self.assertEqual(Rational(0, 5).evaluate(), Int(0))
        # sign normalization
        self.assertEqual(Rational(-4, -6).evaluate(), Rational(2, 3))
        self.assertEqual(Rational(4, -2).evaluate(), Int(-2))
        # integral results
        self.assertEqual(Rational(1, 1).evaluate(), Int(1))
        self.assertEqual(Rational(4, 2).evaluate(), Int(2))
        self.assertEqual(Rational(-4, 2).evaluate(), Int(-2))
        # already reduced fractions are kept
        self.assertEqual(Rational(1, 2).evaluate(), Rational(1, 2))
        self.assertEqual(Rational(-1, 2).evaluate(), Rational(-1, 2))

    def test_complex(self):
        self.assertEqual(S(1 + 2j).evaluate(), Complex(Int(1), Int(2)))
        self.assertEqual(S(0j).evaluate(), Int(0))
        # zero imaginary part collapses to the real part
        self.assertEqual(Complex(Int(1), Int(0)).evaluate(), Int(1))
        self.assertEqual(Complex(Rational(2, 4), Int(0)).evaluate(), Rational(1, 2))
        # real and imaginary parts are normalized independently
        self.assertEqual(
            Complex(Rational(2, 4), Rational(-1, 2)).evaluate(),
            Complex(Rational(1, 2), Rational(-1, 2)),
        )

    def test_as_expr_conversions(self):
        self.assertEqual(S(2), Int(2))
        self.assertEqual(S(-1), Int(-1))
        self.assertEqual(S(2.0), Int(2))
        self.assertEqual(S(2.5), Float(2.5))
        self.assertEqual(S(1 + 2j), Complex(Int(1), Int(2)))
        self.assertEqual(S("x"), Symbol(("root", "x")))
        self.assertRaises(ValueError, S, object())

class PlusEvaluationTests(TestCase):
    def test_constant_folding(self):
        self.assertEqual((S(2) + S(3)).evaluate(), Int(5))
        self.assertEqual((S(3) + S(4) + S(5)).evaluate(), Int(12))
        self.assertEqual((S(2) + S(2.5)).evaluate(), Float(4.5))
        self.assertEqual((Rational(1, 2) + Rational(1, 3)).evaluate(), Rational(5, 6))
        self.assertEqual((S(1) / S(2) + S(1) / S(3)).evaluate(), Rational(5, 6))
        self.assertEqual((Rational(1, 2) + S(0.5)).evaluate(), Int(1))

    def test_like_term_collection(self):
        self.assertEqual((x + x).evaluate(), Times((Int(2), x)))
        self.assertEqual((x + x + x).evaluate(), Times((Int(3), x)))
        self.assertEqual((x + 2 * x).evaluate(), Times((Int(3), x)))
        self.assertEqual((2 * x + 3 * x + 4 * x).evaluate(), Times((Int(9), x)))
        self.assertEqual((x + y + x).evaluate(), Plus((y, Times((Int(2), x)))))
        self.assertEqual(
            (2 * x + 3 * y + S(0.5) * x).evaluate(),
            Plus((Times((Int(3), y)), Times((Float(2.5), x)))),
        )

    def test_zero_cancellation(self):
        self.assertEqual((x - x).evaluate(), Int(0))
        self.assertEqual((2 * x - x).evaluate(), x)
        self.assertEqual((x + 0).evaluate(), x)
        self.assertEqual((0 + x).evaluate(), x)
        self.assertEqual((x + y - x).evaluate(), y)
        self.assertEqual((x + y - x - y).evaluate(), Int(0))

    def test_constant_terms(self):
        self.assertEqual((x + 2).evaluate(), Plus((Int(2), x)))
        self.assertEqual((x + 1 + x).evaluate(), Plus((Int(1), Times((Int(2), x)))))
        self.assertEqual((x + 2 + y).evaluate(), Plus((Int(2), x, y)))

    def test_nested_plus_flattening(self):
        self.assertEqual(((x + y) + (z + x)).evaluate(), Plus((y, z, Times((Int(2), x)))))

    def test_terms_are_sorted(self):
        self.assertEqual((y + x).evaluate(), Plus((x, y)))
        self.assertEqual((x + y + z).evaluate(), Plus((x, y, z)))
        self.assertEqual((z + x).evaluate(), Plus((x, z)))

    def test_symbols_with_complex_constant(self):
        self.assertEqual((x + S(1 + 2j)).evaluate(), Plus((Complex(Int(1), Int(2)), x)))

class TimesEvaluationTests(TestCase):
    def test_constant_folding(self):
        self.assertEqual((S(2) * S(3)).evaluate(), Int(6))
        self.assertEqual((S(2) * x * S(3)).evaluate(), Times((Int(6), x)))
        self.assertEqual(((2 * x) * (3 * y)).evaluate(), Times((Int(6), x, y)))

    def test_power_collection(self):
        self.assertEqual((x * x).evaluate(), Power(x, Int(2)))
        self.assertEqual((x * x * x).evaluate(), Power(x, Int(3)))
        self.assertEqual((S(2) * x * x).evaluate(), Times((Int(2), Power(x, Int(2)))))
        self.assertEqual((x * y * x).evaluate(), Times((y, Power(x, Int(2)))))
        self.assertEqual((x * x * y * x).evaluate(), Times((y, Power(x, Int(3)))))

    def test_zero_and_one(self):
        self.assertEqual((x * 0).evaluate(), Int(0))
        self.assertEqual((0 * x).evaluate(), Int(0))
        self.assertEqual((2 * x * 0).evaluate(), Int(0))
        self.assertEqual((x * 1).evaluate(), x)
        self.assertEqual((x * 1 * 1).evaluate(), x)

    def test_reciprocal(self):
        self.assertEqual((S(2) / x).evaluate(), Times((Int(2), Power(x, Int(-1)))))
        self.assertEqual((x / y).evaluate(), Times((x, Power(y, Int(-1)))))
        self.assertEqual((S(1) / (x * y)).evaluate(), Times((Power(x, Int(-1)), Power(y, Int(-1)))))
        self.assertEqual((x / x).evaluate(), Int(1))

    def test_float_factors(self):
        self.assertEqual((S(2.5) * x).evaluate(), Times((Float(2.5), x)))
        self.assertEqual((S(0.5) * x).evaluate(), Times((Float(0.5), x)))
        self.assertEqual((S(2.5) * S(2) * x).evaluate(), Times((Int(5), x)))

    def test_distribution_is_not_expanded(self):
        # products of sums are kept factored
        self.assertEqual((2 * (x + y)).evaluate(), Times((Int(2), Plus((x, y)))))
        self.assertEqual((x * (y + 1)).evaluate(), Times((x, Plus((Int(1), y)))))
        self.assertEqual(((x + 1) * (x - 1)).evaluate(), Times((Plus((Int(-1), x)), Plus((Int(1), x)))))

class PowerEvaluationTests(TestCase):
    def test_zero_one_exponent(self):
        self.assertEqual((x ** 0).evaluate(), Int(1))
        self.assertEqual(Power.make(x, Int(0)), Int(1))
        self.assertEqual(Power.make(x, Int(1)), x)
        self.assertEqual(Power.make(x, Int(2)), Power(x, Int(2)))

    def test_constant_base(self):
        self.assertEqual((S(2) ** 3).evaluate(), Int(8))
        self.assertEqual((S(2) ** -1).evaluate(), Rational(1, 2))
        self.assertEqual((S(2) ** 0).evaluate(), Int(1))
        self.assertEqual((S(0) ** 5).evaluate(), Int(0))
        self.assertEqual((S(2.0) ** 2).evaluate(), Int(4))
        self.assertEqual((S(2.5) ** 2).evaluate(), Float(6.25))
        self.assertEqual((Rational(1, 2) ** 3).evaluate(), Rational(1, 8))
        self.assertEqual((Rational(1, 2) ** -2).evaluate(), Int(4))

    def test_symbol_base(self):
        self.assertEqual((x ** 2).evaluate(), Power(x, Int(2)))
        self.assertEqual((x ** -2).evaluate(), Power(x, Int(-2)))
        self.assertEqual((x ** 0.5).evaluate(), Power(x, Float(0.5)))
        self.assertEqual((x ** Rational(1, 2)).evaluate(), Power(x, Rational(1, 2)))
        self.assertEqual(x.sqrt().evaluate(), Power(x, Rational(1, 2)))

    def test_times_base_is_distributed(self):
        self.assertEqual(((2 * x) ** 2).evaluate(), Times((Int(4), Power(x, Int(2)))))
        self.assertEqual(((x / 2) ** 2).evaluate(), Times((Rational(1, 4), Power(x, Int(2)))))
        # sums are not expanded
        self.assertEqual(((x + y) ** 2).evaluate(), Power(Plus((x, y)), Int(2)))
        self.assertEqual(((x + y) * (x + y)).evaluate(), Power(Plus((x, y)), Int(2)))

class ComplexArithmeticTests(TestCase):
    def test_add_sub(self):
        self.assertEqual((S(1 + 2j) + S(3 + 4j)).evaluate(), Complex(Int(4), Int(6)))
        self.assertEqual((S(1 + 2j) - S(3 + 4j)).evaluate(), Complex(Int(-2), Int(-2)))
        self.assertEqual((S(1 + 2j) + S(0.5)).evaluate(), Complex(Float(1.5), Int(2)))
        self.assertEqual((S(1 + 2j) + Rational(1, 2)).evaluate(), Complex(Rational(3, 2), Int(2)))

    def test_mul(self):
        self.assertEqual((S(1 + 2j) * S(3 + 4j)).evaluate(), Complex(Int(-5), Int(10)))
        self.assertEqual((S(1 + 2j) * S(1 - 2j)).evaluate(), Int(5))
        self.assertEqual((S(1 + 2j) * S(2)).evaluate(), Complex(Int(2), Int(4)))
        self.assertEqual((S(2) * S(1 + 2j)).evaluate(), Complex(Int(2), Int(4)))
        self.assertEqual((S(1 + 2j) * S(0)).evaluate(), Int(0))
        self.assertEqual((S(1 + 2j) / S(2)).evaluate(), Complex(Rational(1, 2), Int(1)))

    def test_pow(self):
        self.assertEqual((S(1 + 2j) ** 2).evaluate(), Complex(Int(-3), Int(4)))
        self.assertEqual((S(1 + 2j) ** 0).evaluate(), Int(1))
        self.assertEqual((S(1 + 2j) ** -1).evaluate(), Complex(Rational(1, 5), Rational(-2, 5)))

    def test_neg(self):
        self.assertEqual(Complex(Int(1), Int(2)).const_neg(), Complex(Int(-1), Int(-2)))

class MixedArithmeticTests(TestCase):
    def test_int_float_mix(self):
        self.assertEqual((S(1) + S(2.5)).evaluate(), Float(3.5))
        self.assertEqual((S(2) * S(0.5)).evaluate(), Int(1))
        self.assertEqual((S(0.5) * S(2)).evaluate(), Int(1))

    def test_rational_mix(self):
        self.assertEqual((Rational(1, 2) * S(0.5)).evaluate(), Float(0.25))
        self.assertEqual((Rational(1, 2) + S(1)).evaluate(), Rational(3, 2))
        # multiplication by an integer must preserve the denominator
        self.assertEqual((Rational(1, 2) * S(2)).evaluate(), Int(1))
        self.assertEqual((S(2) * Rational(1, 2)).evaluate(), Int(1))
        self.assertEqual((S(1) / S(2) * S(2)).evaluate(), Int(1))
        self.assertEqual((S(2) * (S(1) / S(2))).evaluate(), Int(1))
        self.assertEqual((Rational(2, 3) * S(6)).evaluate(), Int(4))
        self.assertEqual((S(6) * Rational(2, 3)).evaluate(), Int(4))

class ExpressionFormTests(TestCase):
    def test_evaluate_is_idempotent(self):
        exprs = [
            x + 2 * x + y,
            (x + y) * (x - y),
            S(2) * x * x,
            x ** 3 + Rational(1, 2) * x,
            (S(1) + x) / S(2),
            S(1 + 2j) * x + S(3),
        ]
        for e in exprs:
            once = e.evaluate()
            self.assertEqual(once, once.evaluate(), f"evaluate is not idempotent for {e}")

    def test_evaluate_accepts_a_context(self):
        # Context is a plain marker for now; evaluation must accept it unchanged
        ctx = Context()
        self.assertEqual((x + x).evaluate(ctx), Times((Int(2), x)))
        self.assertEqual(S(1 + 2j).evaluate(ctx), Complex(Int(1), Int(2)))

    def test_unevaluable_forms_are_preserved(self):
        self.assertEqual(Sin(x).evaluate(), Sin(x))
        self.assertEqual(Cos(S(0)).evaluate(), Cos(Int(0)))
        self.assertEqual(Ln(x).evaluate(), Ln(x))
        self.assertEqual(Exp(x).evaluate(), Exp(x))
        self.assertEqual(Roll(x, 0, 1).evaluate(), Roll(x, 0, 1))
        self.assertEqual(Slice(x, 0, 2).evaluate(), Slice(x, 0, 2))
        self.assertEqual(SymbolShape(x, 0).evaluate(), SymbolShape(x, 0))

    def test_symbols_are_distinct_by_name(self):
        a, b = symbols('a', 'b')
        self.assertNotEqual(a, b)
        self.assertEqual(a, a)
        # symbol('a') and S('a') differ in their name prefix
        self.assertNotEqual(a, S('a'))

all_tests = [
    ConstantEvaluationTests,
    PlusEvaluationTests,
    TimesEvaluationTests,
    PowerEvaluationTests,
    ComplexArithmeticTests,
    MixedArithmeticTests,
    ExpressionFormTests,
]
