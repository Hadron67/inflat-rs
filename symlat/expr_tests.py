from unittest import TestCase

from .expr import (
    Complex,
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
    Times,
    symbols,
)

x, y, z = symbols('x', 'y', 'z')

class ConstantEvaluationTests(TestCase):
    def test_int(self):
        self.assertEqual(Int(3).normalize(), Int(3))
        self.assertEqual(Int(0).normalize(), Int(0))
        self.assertEqual(Int(-7).normalize(), Int(-7))

    def test_float(self):
        # integral floats are normalized to Int
        self.assertEqual(S(2.0).normalize(), Int(2))
        self.assertEqual(S(0.0).normalize(), Int(0))
        self.assertEqual(S(-3.0).normalize(), Int(-3))
        # non-integral floats are kept
        self.assertEqual(S(2.5).normalize(), Float(2.5))
        self.assertEqual(S(-0.5).normalize(), Float(-0.5))

    def test_rational(self):
        # fraction reduction
        self.assertEqual(Rational(6, 4).normalize(), Rational(3, 2))
        self.assertEqual(Rational(2, 4).normalize(), Rational(1, 2))
        # zero numerator
        self.assertEqual(Rational(0, 5).normalize(), Int(0))
        # sign normalization
        self.assertEqual(Rational(-4, -6).normalize(), Rational(2, 3))
        self.assertEqual(Rational(4, -2).normalize(), Int(-2))
        # integral results
        self.assertEqual(Rational(1, 1).normalize(), Int(1))
        self.assertEqual(Rational(4, 2).normalize(), Int(2))
        self.assertEqual(Rational(-4, 2).normalize(), Int(-2))
        # already reduced fractions are kept
        self.assertEqual(Rational(1, 2).normalize(), Rational(1, 2))
        self.assertEqual(Rational(-1, 2).normalize(), Rational(-1, 2))

    def test_complex(self):
        self.assertEqual(S(1 + 2j).normalize(), Complex(Int(1), Int(2)))
        self.assertEqual(S(0j).normalize(), Int(0))
        # zero imaginary part collapses to the real part
        self.assertEqual(Complex(Int(1), Int(0)).normalize(), Int(1))
        self.assertEqual(Complex(Rational(2, 4), Int(0)).normalize(), Rational(1, 2))
        # real and imaginary parts are normalized independently
        self.assertEqual(
            Complex(Rational(2, 4), Rational(-1, 2)).normalize(),
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
        self.assertEqual((S(2) + S(3)).normalize(), Int(5))
        self.assertEqual((S(3) + S(4) + S(5)).normalize(), Int(12))
        self.assertEqual((S(2) + S(2.5)).normalize(), Float(4.5))
        self.assertEqual((Rational(1, 2) + Rational(1, 3)).normalize(), Rational(5, 6))
        self.assertEqual((S(1) / S(2) + S(1) / S(3)).normalize(), Rational(5, 6))
        self.assertEqual((Rational(1, 2) + S(0.5)).normalize(), Int(1))

    def test_like_term_collection(self):
        self.assertEqual((x + x).normalize(), Times((Int(2), x)))
        self.assertEqual((x + x + x).normalize(), Times((Int(3), x)))
        self.assertEqual((x + 2 * x).normalize(), Times((Int(3), x)))
        self.assertEqual((2 * x + 3 * x + 4 * x).normalize(), Times((Int(9), x)))
        self.assertEqual((x + y + x).normalize(), Plus((y, Times((Int(2), x)))))
        self.assertEqual(
            (2 * x + 3 * y + S(0.5) * x).normalize(),
            Plus((Times((Int(3), y)), Times((Float(2.5), x)))),
        )

    def test_zero_cancellation(self):
        self.assertEqual((x - x).normalize(), Int(0))
        self.assertEqual((2 * x - x).normalize(), x)
        self.assertEqual((x + 0).normalize(), x)
        self.assertEqual((0 + x).normalize(), x)
        self.assertEqual((x + y - x).normalize(), y)
        self.assertEqual((x + y - x - y).normalize(), Int(0))

    def test_constant_terms(self):
        self.assertEqual((x + 2).normalize(), Plus((Int(2), x)))
        self.assertEqual((x + 1 + x).normalize(), Plus((Int(1), Times((Int(2), x)))))
        self.assertEqual((x + 2 + y).normalize(), Plus((Int(2), x, y)))

    def test_nested_plus_flattening(self):
        self.assertEqual(((x + y) + (z + x)).normalize(), Plus((y, z, Times((Int(2), x)))))

    def test_terms_are_sorted(self):
        self.assertEqual((y + x).normalize(), Plus((x, y)))
        self.assertEqual((x + y + z).normalize(), Plus((x, y, z)))
        self.assertEqual((z + x).normalize(), Plus((x, z)))

    def test_symbols_with_complex_constant(self):
        self.assertEqual((x + S(1 + 2j)).normalize(), Plus((Complex(Int(1), Int(2)), x)))

class TimesEvaluationTests(TestCase):
    def test_constant_folding(self):
        self.assertEqual((S(2) * S(3)).normalize(), Int(6))
        self.assertEqual((S(2) * x * S(3)).normalize(), Times((Int(6), x)))
        self.assertEqual(((2 * x) * (3 * y)).normalize(), Times((Int(6), x, y)))

    def test_power_collection(self):
        self.assertEqual((x * x).normalize(), Power(x, Int(2)))
        self.assertEqual((x * x * x).normalize(), Power(x, Int(3)))
        self.assertEqual((S(2) * x * x).normalize(), Times((Int(2), Power(x, Int(2)))))
        self.assertEqual((x * y * x).normalize(), Times((y, Power(x, Int(2)))))
        self.assertEqual((x * x * y * x).normalize(), Times((y, Power(x, Int(3)))))

    def test_zero_and_one(self):
        self.assertEqual((x * 0).normalize(), Int(0))
        self.assertEqual((0 * x).normalize(), Int(0))
        self.assertEqual((2 * x * 0).normalize(), Int(0))
        self.assertEqual((x * 1).normalize(), x)
        self.assertEqual((x * 1 * 1).normalize(), x)

    def test_reciprocal(self):
        self.assertEqual((S(2) / x).normalize(), Times((Int(2), Power(x, Int(-1)))))
        self.assertEqual((x / y).normalize(), Times((x, Power(y, Int(-1)))))
        self.assertEqual((S(1) / (x * y)).normalize(), Times((Power(x, Int(-1)), Power(y, Int(-1)))))
        self.assertEqual((x / x).normalize(), Int(1))

    def test_float_factors(self):
        self.assertEqual((S(2.5) * x).normalize(), Times((Float(2.5), x)))
        self.assertEqual((S(0.5) * x).normalize(), Times((Float(0.5), x)))
        self.assertEqual((S(2.5) * S(2) * x).normalize(), Times((Int(5), x)))

    def test_distribution_is_not_expanded(self):
        # products of sums are kept factored
        self.assertEqual((2 * (x + y)).normalize(), Times((Int(2), Plus((x, y)))))
        self.assertEqual((x * (y + 1)).normalize(), Times((x, Plus((Int(1), y)))))
        self.assertEqual(((x + 1) * (x - 1)).normalize(), Times((Plus((Int(-1), x)), Plus((Int(1), x)))))

class PowerEvaluationTests(TestCase):
    def test_zero_one_exponent(self):
        self.assertEqual((x ** 0).normalize(), Int(1))
        self.assertEqual(Power.make(x, Int(0)), Int(1))
        self.assertEqual(Power.make(x, Int(1)), x)
        self.assertEqual(Power.make(x, Int(2)), Power(x, Int(2)))

    def test_constant_base(self):
        self.assertEqual((S(2) ** 3).normalize(), Int(8))
        self.assertEqual((S(2) ** -1).normalize(), Rational(1, 2))
        self.assertEqual((S(2) ** 0).normalize(), Int(1))
        self.assertEqual((S(0) ** 5).normalize(), Int(0))
        self.assertEqual((S(2.0) ** 2).normalize(), Int(4))
        self.assertEqual((S(2.5) ** 2).normalize(), Float(6.25))
        self.assertEqual((Rational(1, 2) ** 3).normalize(), Rational(1, 8))
        self.assertEqual((Rational(1, 2) ** -2).normalize(), Int(4))

    def test_symbol_base(self):
        self.assertEqual((x ** 2).normalize(), Power(x, Int(2)))
        self.assertEqual((x ** -2).normalize(), Power(x, Int(-2)))
        self.assertEqual((x ** 0.5).normalize(), Power(x, Float(0.5)))
        self.assertEqual((x ** Rational(1, 2)).normalize(), Power(x, Rational(1, 2)))
        self.assertEqual(x.sqrt().normalize(), Power(x, Rational(1, 2)))

    def test_times_base_is_distributed(self):
        self.assertEqual(((2 * x) ** 2).normalize(), Times((Int(4), Power(x, Int(2)))))
        self.assertEqual(((x / 2) ** 2).normalize(), Times((Rational(1, 4), Power(x, Int(2)))))
        # sums are not expanded
        self.assertEqual(((x + y) ** 2).normalize(), Power(Plus((x, y)), Int(2)))
        self.assertEqual(((x + y) * (x + y)).normalize(), Power(Plus((x, y)), Int(2)))

class ComplexArithmeticTests(TestCase):
    def test_add_sub(self):
        self.assertEqual((S(1 + 2j) + S(3 + 4j)).normalize(), Complex(Int(4), Int(6)))
        self.assertEqual((S(1 + 2j) - S(3 + 4j)).normalize(), Complex(Int(-2), Int(-2)))
        self.assertEqual((S(1 + 2j) + S(0.5)).normalize(), Complex(Float(1.5), Int(2)))
        self.assertEqual((S(1 + 2j) + Rational(1, 2)).normalize(), Complex(Rational(3, 2), Int(2)))

    def test_mul(self):
        self.assertEqual((S(1 + 2j) * S(3 + 4j)).normalize(), Complex(Int(-5), Int(10)))
        self.assertEqual((S(1 + 2j) * S(1 - 2j)).normalize(), Int(5))
        self.assertEqual((S(1 + 2j) * S(2)).normalize(), Complex(Int(2), Int(4)))
        self.assertEqual((S(2) * S(1 + 2j)).normalize(), Complex(Int(2), Int(4)))
        self.assertEqual((S(1 + 2j) * S(0)).normalize(), Int(0))
        self.assertEqual((S(1 + 2j) / S(2)).normalize(), Complex(Rational(1, 2), Int(1)))

    def test_pow(self):
        self.assertEqual((S(1 + 2j) ** 2).normalize(), Complex(Int(-3), Int(4)))
        self.assertEqual((S(1 + 2j) ** 0).normalize(), Int(1))
        self.assertEqual((S(1 + 2j) ** -1).normalize(), Complex(Rational(1, 5), Rational(-2, 5)))

    def test_neg(self):
        self.assertEqual(Complex(Int(1), Int(2)).const_neg(), Complex(Int(-1), Int(-2)))

class MixedArithmeticTests(TestCase):
    def test_int_float_mix(self):
        self.assertEqual((S(1) + S(2.5)).normalize(), Float(3.5))
        self.assertEqual((S(2) * S(0.5)).normalize(), Int(1))
        self.assertEqual((S(0.5) * S(2)).normalize(), Int(1))

    def test_rational_mix(self):
        self.assertEqual((Rational(1, 2) * S(0.5)).normalize(), Float(0.25))
        self.assertEqual((Rational(1, 2) + S(1)).normalize(), Rational(3, 2))
        # multiplication by an integer must preserve the denominator
        self.assertEqual((Rational(1, 2) * S(2)).normalize(), Int(1))
        self.assertEqual((S(2) * Rational(1, 2)).normalize(), Int(1))
        self.assertEqual((S(1) / S(2) * S(2)).normalize(), Int(1))
        self.assertEqual((S(2) * (S(1) / S(2))).normalize(), Int(1))
        self.assertEqual((Rational(2, 3) * S(6)).normalize(), Int(4))
        self.assertEqual((S(6) * Rational(2, 3)).normalize(), Int(4))

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
            once = e.normalize()
            self.assertEqual(once, once.normalize(), f"evaluate is not idempotent for {e}")

    def test_unevaluable_forms_are_preserved(self):
        self.assertEqual(Sin(x).normalize(), Sin(x))
        self.assertEqual(Cos(S(0)).normalize(), Cos(Int(0)))
        self.assertEqual(Ln(x).normalize(), Ln(x))
        self.assertEqual(Exp(x).normalize(), Exp(x))
        self.assertEqual(Roll(x, 0, 1).normalize(), Roll(x, 0, 1))
        self.assertEqual(Slice(x, 0, 2).normalize(), Slice(x, 0, 2))

    def test_symbols_are_distinct_by_name(self):
        a, b = symbols('a', 'b')
        self.assertNotEqual(a, b)
        self.assertEqual(a, a)
        # symbol('a') and S('a') differ in their name prefix
        self.assertNotEqual(a, S('a'))

class ComplexExpressionNormalizationTests(TestCase):
    """Normalization of larger expressions combining multiple features."""

    def test_polynomial(self):
        self.assertEqual(
            (x ** 3 + 3 * x ** 2 * y + 3 * x * y ** 2 + y ** 3).normalize(),
            Plus((
                Times((Int(3), x, Power(y, Int(2)))),
                Times((Int(3), y, Power(x, Int(2)))),
                Power(x, Int(3)),
                Power(y, Int(3)),
            )),
        )

    def test_like_term_collection_across_structures(self):
        # powers are collected like any other factor
        self.assertEqual((x ** 2 + x ** 2 + x ** 2).normalize(), Times((Int(3), Power(x, Int(2)))))
        # like terms are recognized regardless of factor order
        self.assertEqual((x ** 2 * y + y * x ** 2).normalize(), Times((Int(2), y, Power(x, Int(2)))))
        self.assertEqual((x ** 2 * y ** 3 + x ** 2 * y ** 3).normalize(), Times((Int(2), Power(x, Int(2)), Power(y, Int(3)))))
        self.assertEqual(
            (x ** 2 * y + x * y ** 2 + x ** 2 * y).normalize(),
            Plus((Times((Int(2), y, Power(x, Int(2)))), Times((x, Power(y, Int(2)))))),
        )
        # rational coefficients are summed exactly
        self.assertEqual(
            (Rational(1, 2) * x ** 2 + Rational(1, 3) * x ** 2).normalize(),
            Times((Rational(5, 6), Power(x, Int(2)))),
        )

    def test_power_of_product_distributes(self):
        self.assertEqual(
            ((x * y) ** 2 + x ** 2 * y ** 2).normalize(),
            Times((Int(2), Power(x, Int(2)), Power(y, Int(2)))),
        )

    def test_products_of_sums_are_kept_factored(self):
        self.assertEqual(
            ((2 * x + 3 * y) * (4 * x - 5 * y)).normalize(),
            Times((
                Plus((Times((Int(-5), y)), Times((Int(4), x)))),
                Plus((Times((Int(2), x)), Times((Int(3), y)))),
            )),
        )
        self.assertEqual(
            ((x + 2) * (x + 3) * (x + 4)).normalize(),
            Times((Plus((Int(2), x)), Plus((Int(3), x)), Plus((Int(4), x)))),
        )

    def test_repeated_factors_are_collected_into_powers(self):
        self.assertEqual(((x + y) * (x + y) * (x + y)).normalize(), Power(Plus((x, y)), Int(3)))
        self.assertEqual(
            ((x + y) * (x - y) * (x + y)).normalize(),
            Times((Plus((x, Times((Int(-1), y)))), Power(Plus((x, y)), Int(2)))),
        )
        self.assertEqual(
            (((x + y) * (x - y)) ** 2).normalize(),
            Times((Power(Plus((x, y)), Int(2)), Power(Plus((x, Times((Int(-1), y)))), Int(2)))),
        )

    def test_fractional_expressions(self):
        self.assertEqual(
            (x / y + y / x).normalize(),
            Plus((Times((x, Power(y, Int(-1)))), Times((y, Power(x, Int(-1)))))),
        )
        self.assertEqual((S(1) / (x + y)).normalize(), Power(Plus((x, y)), Int(-1)))
        self.assertEqual(
            ((x + 1) / (y + 2)).normalize(),
            Times((Plus((Int(1), x)), Power(Plus((Int(2), y)), Int(-1)))),
        )
        self.assertEqual(((x / y) * (y / x)).normalize(), Int(1))
        self.assertEqual((x / 2 + x / 4 + x / 8).normalize(), Times((Rational(7, 8), x)))

    def test_nested_normalization(self):
        self.assertEqual(
            (((x + 1) + (x + 2)) / S(3)).normalize(),
            Times((Rational(1, 3), Plus((Int(3), Times((Int(2), x)))))),
        )
        # sums that normalize into a Times are flattened into the enclosing Times
        self.assertEqual(((x + x + x) * (y + y)).normalize(), Times((Int(6), x, y)))
        self.assertEqual(((2 * x) * (3 * y) * (4 * z)).normalize(), Times((Int(24), x, y, z)))
        self.assertEqual(
            ((x ** 2 - 1) / (x - 1)).normalize(),
            Times((Plus((Int(-1), Power(x, Int(2)))), Power(Plus((Int(-1), x)), Int(-1)))),
        )

    def test_powers_of_powers_are_combined(self):
        # (a^m)^n = a^(m*n) and x^1 folds to x
        self.assertEqual(((x ** 2) ** 3).normalize(), Power(x, Int(6)))
        self.assertEqual(((x ** 2) ** 0).normalize(), Int(1))
        self.assertEqual(((x ** -1) ** 2).normalize(), Power(x, Int(-2)))
        self.assertEqual((x ** 3 / x).normalize(), Power(x, Int(2)))
        self.assertEqual((x ** 2 / x).normalize(), x)
        self.assertEqual((x / x ** 2).normalize(), Power(x, Int(-1)))
        self.assertEqual(
            (x ** 2 / y ** 2).normalize(),
            Times((Power(x, Int(2)), Power(y, Int(-2)))),
        )
        self.assertEqual(
            ((x / y) ** 2).normalize(),
            Times((Power(x, Int(2)), Power(y, Int(-2)))),
        )
        self.assertEqual(
            (x ** 2 / y ** 3 + 1).normalize(),
            Plus((Int(1), Times((Power(x, Int(2)), Power(y, Int(-3)))))),
        )
        # distribution and exponent combination cooperate
        self.assertEqual((x ** 2 * y ** 2 / (x * y)).normalize(), Times((x, y)))

    def test_complex_coefficients(self):
        self.assertEqual(
            (S(1 + 2j) * (x + y) + S(2 - 1j) * (x + y)).normalize(),
            Times((Complex(Int(3), Int(1)), Plus((x, y)))),
        )
        self.assertEqual(
            (S(1 + 2j) * x + S(3 - 4j) * y + x).normalize(),
            Plus((Times((Complex(Int(2), Int(2)), x)), Times((Complex(Int(3), Int(-4)), y)))),
        )

    def test_function_terms(self):
        # applications of numeric functions take part in like-term collection
        self.assertEqual((Sin(x) + Sin(x)).normalize(), Times((Int(2), Sin(x))))

all_tests = [
    ConstantEvaluationTests,
    PlusEvaluationTests,
    TimesEvaluationTests,
    PowerEvaluationTests,
    ComplexArithmeticTests,
    MixedArithmeticTests,
    ExpressionFormTests,
    ComplexExpressionNormalizationTests,
]
