from abc import abstractmethod
from dataclasses import Field, dataclass, field
from inspect import isclass
from typing import ReadOnly, dataclass_transform, override

from typing_extensions import Callable

class Expr:
    @staticmethod
    def as_expr(expr) -> 'Expr':
        if isinstance(expr, Expr):
            return expr
        match expr:
            case str():
                return Symbol(tuple(['root'] + expr.split('.')))
            case int():
                return Int(expr)
            case float():
                return Float(expr).normalize()
            case complex():
                re = Expr.as_expr(expr.real)
                im = Expr.as_expr(expr.imag)
                assert isinstance(re, Constant) and isinstance(im, Constant)
                return Complex(re, im)
            case _:
                raise ValueError(f"Cannot convert {expr} to Expr")

    def subexpressions(self) -> 'list[Expr]':
        return []

    @abstractmethod
    def with_subexpressions(self, children: 'list[Expr]') -> 'Expr':
        raise NotImplementedError

    @abstractmethod
    def head_sort_token(self) -> int:
        pass

    @abstractmethod
    def sort_key(self) -> list[object]:
        pass

    @abstractmethod
    def get_type(self) -> 'Expr':
        raise NotImplementedError

    @abstractmethod
    def compare(self, other: 'Expr') -> int:
        raise NotImplementedError

    @abstractmethod
    def get_shape(self) -> 'tuple[Expr, ...] | None':
        """
            Get the shape of an expression.
            `None` means unspecified
        """
        return None

    def is_subtype(self, other: 'Expr') -> bool:
        return False

    def evaluate(self, context: 'Context | None' = None) -> 'Expr':
        return self

    def __lt__(self, other) -> bool:
        if isinstance(other, Expr):
            return self.compare(other) < 0
        else:
            raise ValueError(f"cannot compare {self} with {other}")

    def __gt__(self, other) -> bool:
        if isinstance(other, Expr):
            return self.compare(other) > 0
        else:
            raise ValueError(f"cannot compare {self} with {other}")

    def __add__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Plus((self, other))

    def __radd__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Plus((other, self))

    def __sub__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Plus((self, Times((Int(-1), other))))

    def __rsub__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Plus((other, Times((Int(-1), self))))

    def __mul__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Times((self, other))

    def __rmul__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Times((other, self))

    def __truediv__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Times((self, Power(other, Int(-1))))

    def __rtruediv__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Times((other, Power(self, Int(-1))))

    def __pow__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Power(self, other)

    def __rpow__(self, other) -> 'Expr':
        other = Expr.as_expr(other)
        return Power(other, self)

    def sqrt(self) -> 'Expr':
        return Power(self, Rational(1, 2))

    @abstractmethod
    def input_form(self) -> str:
        pass

    def is_zero(self) -> bool:
        return False

    def is_one(self) -> bool:
        return False

    def __repr__(self) -> str:
        return self.input_form()

    def separate_constant_coefficient(self) -> tuple['Constant', 'Expr']:
        return Int(1), self

    def as_power(self) -> 'Power':
        return Power(self, Int(1))

    HEAD_SORT_TOKEN: int

    _HEAD_SORT_TOKEN_COUNTER: int = 0

def S(expr) -> Expr:
    return Expr.as_expr(expr)

@dataclass_transform()
def exprclass(cls=None, **kwargs):
    dataclass_wrapper = dataclass(repr=False, frozen=True, **kwargs)
    def wrapper(cls):
        other_fields: list[str] = []
        expr_fields: list[str] = []
        expr_list_field: str | None = None
        for base_expr_class in cls.mro()[-1::-1]:
            if base_expr_class is object or base_expr_class is Expr:
                continue
            for name, type in base_expr_class.__annotations__.items():
                field_data = getattr(base_expr_class, name, None)
                if field_data is not None and isinstance(field_data, Field) and not field_data.init:
                    continue
                if isclass(type) and issubclass(type, Expr):
                    expr_fields.append(name)
                elif type == tuple[Expr, ...]:
                    if expr_list_field is None:
                        expr_list_field = name
                    else:
                        raise ValueError(f"Multiple tuple[Expr, ...] fields found: {expr_list_field} and {name}")
                else:
                    other_fields.append(name)

        def subexpressions(self) -> list[Expr]:
            ret: list[Expr] = []
            for name in expr_fields:
                ret.append(getattr(self, name))
            if expr_list_field is not None:
                ret.extend(getattr(self, expr_list_field))
            return ret

        def with_subexpressions(self, children: list[Expr]) -> Expr:
            ret = self.copy()
            cursor = 0
            for name in expr_fields:
                setattr(ret, name, children[cursor])
                cursor += 1
            if expr_list_field is not None:
                setattr(ret, expr_list_field, children[cursor:])
            return ret

        def sort_key(self) -> list[object]:
            ret = [getattr(cls, 'HEAD_SORT_TOKEN')]
            for name in other_fields:
                ret.append(getattr(self, name))
            for name in expr_fields:
                ret.append(getattr(self, name).sort_key())
            if expr_list_field is not None:
                ret.extend([expr.sort_key() for expr in getattr(self, expr_list_field)])
            return ret

        def cmp(self, other: Expr) -> int:
            if isinstance(other, cls):
                for name in other_fields:
                    field1 = getattr(self, name)
                    field2 = getattr(other, name)
                    if field1 > field2:
                        return 1
                    if field1 < field2:
                        return -1
                for name in expr_fields:
                    if (c := getattr(self, name).compare(getattr(other, name))) != 0:
                        return c
                if expr_list_field is not None:
                    list1 = getattr(self, expr_list_field)
                    list2 = getattr(other, expr_list_field)
                    if len(list1) < len(list2):
                        return -1
                    if len(list1) > len(list2):
                        return 1
                    for expr1, expr2 in zip(list1, list2):
                        if (c := expr1.compare(expr2)) != 0:
                            return c
                return 0
            else:
                token1 = getattr(cls, 'HEAD_SORT_TOKEN')
                token2 = getattr(other, 'HEAD_SORT_TOKEN')
                if token1 > token2:
                    return 1
                if token1 < token2:
                    return -1
                return 0

        Expr._HEAD_SORT_TOKEN_COUNTER += 1
        cls = dataclass_wrapper(cls)
        setattr(cls, "with_subexpressions", with_subexpressions)
        setattr(cls, "subexpressions", subexpressions)
        setattr(cls, "sort_key", sort_key)
        setattr(cls, "compare", cmp)
        setattr(cls, "HEAD_SORT_TOKEN", Expr._HEAD_SORT_TOKEN_COUNTER)
        return cls

    if cls is not None:
        return wrapper(cls)

    return wrapper

TMP_SYMBOL_ROOT = "tmp"


@exprclass
class Universe(Expr):
    level: int

    @override
    def get_type(self) -> Expr:
        return Universe(self.level + 1)

@exprclass
class Type0(Expr):
    def get_type(self) -> Expr:
        return Universe(0)

@exprclass
class IntegerType(Type0):
    @override
    def is_subtype(self, other: Expr) -> bool:
        match other:
            case IntegerType():
                return True
            case _:
                return False

@exprclass
class RealType(Type0):
    @override
    def is_subtype(self, other: Expr) -> bool:
        match other:
            case IntegerType() | RealType():
                return True
            case _:
                return False

@exprclass
class ComplexType(Type0):
    @override
    def is_subtype(self, other: Expr) -> bool:
        match other:
            case IntegerType() | RealType() | ComplexType():
                return True
            case _:
                return False

def merge_shape(shape1: tuple[Expr, ...] | None, shape2: tuple[Expr, ...] | None) -> tuple[Expr, ...] | None:
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1
    if len(shape1) == len(shape2):
        assert shape1 == shape2, f"shape mismatch {shape1} != {shape2}"
        return shape1
    ret: list[Expr] = []
    for i in range(max(len(shape1), len(shape2))):
        if i < len(shape1) and i < len(shape2):
            s1 = shape1[-1 - i]
            s2 = shape2[-1 - i]
            assert s1 == s2, f"incompatible shapes {shape1} != {shape2}"
            ret.append(s1)
        elif i < len(shape1):
            ret.append(shape1[-1 - i])
        elif i < len(shape2):
            ret.append(shape2[-1 - i])
    ret.reverse()
    return tuple(ret)

def merge_shapes(shapes: tuple[tuple[Expr, ...] | None, ...]) -> tuple[Expr, ...] | None:
    assert len(shapes) > 0
    ret = shapes[0]
    for shape in shapes[1:]:
        ret = merge_shape(ret, shape)
    return ret

class Context:
    pass

class Constant(Expr):
    @abstractmethod
    def normalize(self) -> 'Constant':
        pass

    @abstractmethod
    def const_add(self, other: 'Constant') -> 'Constant':
        pass

    @abstractmethod
    def const_mul(self, other: 'Constant') -> 'Constant':
        pass

    @abstractmethod
    def const_inverse(self) -> 'Constant':
        pass

    @abstractmethod
    def const_neg(self) -> 'Constant':
        pass

    def int_pow(self, other: int) -> 'Constant':
        if other == 0:
            return Int(1)
        if other == 1:
            return self
        neg = False
        if other < 0:
            other = -other
            neg = True
        ret = self
        for _ in range(other - 1):
            ret = ret.const_mul(self)
        if neg:
            return ret.const_inverse()
        return ret

    def __pow__(self, other) -> 'Expr':
        if type(other) is int:
            return self.int_pow(other)
        other = Expr.as_expr(other)
        if isinstance(other, Int):
            return self.int_pow(other.value)
        return super().__pow__(other)

    def evaluate(self, context: 'Context | None' = None) -> 'Expr':
        return self.normalize()

class PrimitiveConstant(Constant):
    pass

@exprclass
class Int(PrimitiveConstant):
    value: int

    @override
    def normalize(self) -> 'Constant':
        return self

    @override
    def input_form(self) -> str:
        return str(self.value)

    @override
    def is_zero(self) -> bool:
        return self.value == 0

    @override
    def is_one(self) -> bool:
        return self.value == 1

    @override
    def evaluate(self, context: 'Context | None' = None) -> 'Expr':
        return self

    @override
    def const_inverse(self) -> 'Constant':
        if self.value == 1:
            return self
        return Rational(1, self.value)

    @override
    def const_neg(self) -> 'Constant':
        if self.value == 0:
            return self
        return Int(-self.value)

    @override
    def const_add(self, other: 'Constant') -> 'Constant':
        match other:
            case Int(value):
                return Int(self.value + value)
            case Float(value):
                return Float(self.value + value)
            case Rational(numerator, denominator):
                return Rational(self.value * denominator + numerator, denominator)
            case Complex():
                return other.const_add(self)
        return super().const_add(other)

    @override
    def const_mul(self, other: 'Constant') -> 'Constant':
        match other:
            case Int(value):
                return Int(self.value * value)
            case Float(value):
                return Float(self.value * value)
            case Rational(numerator, denominator):
                return Rational(self.value * numerator, denominator)
            case Complex():
                return other.const_mul(self)
        return super().const_mul(other)

    @override
    def get_type(self) -> Expr:
        return IntegerType()

@exprclass
class Float(PrimitiveConstant):
    value: float

    @override
    def input_form(self) -> str:
        return str(self.value)

    @override
    def is_zero(self) -> bool:
        return self.value == 0

    @override
    def normalize(self) -> 'Constant':
        if self.value == 0:
            return Int(0)
        if self.value.is_integer():
            return Int(int(self.value))
        return self

    @override
    def const_inverse(self) -> 'Constant':
        return Float(1.0 / self.value)

    @override
    def const_neg(self) -> 'Constant':
        return Float(-self.value)

    @override
    def const_add(self, other: 'Constant') -> 'Constant':
        match other:
            case Float(value):
                return Float(self.value + value)
            case Int(value):
                return Float(self.value + value)
            case Rational(numerator, denominator):
                return Float(self.value + numerator / denominator)
            case Complex():
                return other.const_add(self)
        return super().const_add(other)

    @override
    def const_mul(self, other: 'Constant') -> 'Constant':
        match other:
            case Float(value):
                return Float(self.value * value)
            case Int(value):
                return Float(self.value * value)
            case Rational(numerator, denominator):
                return Float(self.value * numerator / denominator)
            case Complex():
                return other.const_mul(self)
        return super().const_mul(other)

    @override
    def get_type(self) -> Expr:
        return RealType()

    def get_shape(self) -> tuple[Expr, ...] | None:
        return ()

@exprclass
class Rational(PrimitiveConstant):
    numerator: int
    denominator: int

    @override
    def input_form(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @override
    def is_zero(self) -> bool:
        return self.numerator == 0

    @override
    def normalize(self) -> 'Constant':
        if self.numerator == 0:
            return Int(0)
        num = self.numerator
        den = self.denominator
        if den < 0:
            num = -num
            den = -den

        sign = 1
        if num < 0:
            sign = -1
            num = -num

        gcd = _gcd(num, den)
        num //= gcd
        den //= gcd

        num *= sign
        if num == den:
            return Int(sign)
        return Rational(num, den) if num != self.numerator or den != self.denominator else self

    @override
    def const_inverse(self) -> 'Constant':
        return Rational(self.denominator, self.numerator)

    @override
    def const_neg(self) -> 'Constant':
        return Rational(-self.numerator, self.denominator)

    @override
    def const_add(self, other: 'Constant') -> 'Constant':
        match other:
            case Rational(numerator, denominator):
                return Rational(self.numerator * denominator + numerator * self.denominator, self.denominator * denominator)
            case Int(value):
                return Rational(self.numerator + value * self.denominator, self.denominator)
            case Float(value):
                return Float(self.numerator / self.denominator + value)
            case Complex():
                return other.const_add(self)
        return super().const_add(other)

    @override
    def const_mul(self, other: 'Constant') -> 'Constant':
        match other:
            case Rational(numerator, denominator):
                return Rational(self.numerator * numerator, self.denominator * denominator)
            case Int(value):
                return Int(self.numerator * value)
            case Float(value):
                return Float(self.numerator * value / self.denominator)
            case Complex():
                return other.const_mul(self)
        return super().const_mul(other)

    @override
    def get_type(self) -> Expr:
        return RealType()

@exprclass
class Complex(Constant):
    re: Constant
    im: Constant

    @override
    def is_zero(self) -> bool:
        return self.re.is_zero() and self.im.is_zero()

    @override
    def normalize(self) -> 'Constant':
        re = self.re.normalize()
        im = self.im.normalize()
        if im.is_zero():
            return re
        return Complex(re, im) if re != self.re or im != self.im else self

    @override
    def input_form(self) -> str:
        if self.re.is_zero():
            return f"{self.im}i"
        return f"{self.re} + {self.im}i"

    @override
    def const_add(self, other: 'Constant') -> 'Constant':
        match other:
            case Complex(re, im):
                return Complex(self.re.const_add(re), self.im.const_add(im))
            case Int(_) | Float(_) | Rational(_, _):
                return Complex(self.re.const_add(other), self.im)
            case _:
                raise NotImplementedError

    @override
    def const_mul(self, other: 'Constant') -> 'Constant':
        match other:
            case Complex(re, im):
                return Complex(
                    self.re.const_mul(re).const_add(self.im.const_mul(im).const_neg()),
                    self.re.const_mul(im).const_add(self.im.const_mul(re)),
                )
            case Int() | Float() | Rational():
                return Complex(self.re.const_mul(other), self.im.const_mul(other))
            case _:
                raise NotImplementedError

    @override
    def const_neg(self) -> 'Constant':
        return Complex(self.re.const_neg(), self.im.const_neg())

    @override
    def const_inverse(self) -> 'Constant':
        denom = self.re.const_mul(self.re).const_add(self.im.const_mul(self.im)).const_inverse()
        return Complex(self.re.const_mul(denom), self.im.const_mul(denom).const_neg())

def _gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a

@exprclass
class Symbol(Expr):
    fully_qualified_name: tuple[str, ...]
    type: Expr = field(default=RealType(), compare=False)
    shape: tuple[Expr, ...] | None = field(default=None, compare=False)

    def input_form(self) -> str:
        return ".".join(self.fully_qualified_name)

    @override
    def get_type(self) -> Expr:
        return self.type

    def get_shape(self) -> tuple[Expr, ...] | None:
        return self.shape

def symbol(name: str) -> Symbol:
    return Symbol(tuple(name.split(".")), ComplexType())

@exprclass
class Plus(Expr):
    children: tuple[Expr, ...]
    type: Expr = field(init=False, compare=False)
    shape: tuple[Expr, ...] | None = field(init=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'type', peer_type(tuple(x.get_type() for x in self.children)))
        object.__setattr__(self, 'shape', merge_shapes(tuple(i.get_shape() for i in self.children)))

    @abstractmethod
    def get_type(self) -> 'Expr':
        return self.type

    @staticmethod
    def make(children: tuple[Expr, ...]) -> Expr:
        if len(children) == 0:
            return Int(0)
        if len(children) == 1:
            return children[0]
        return Plus(children)

    @staticmethod
    def connect_terms(*start: Expr) -> tuple[Expr, ...]:
        todo = list(start)
        ret: list[Expr] = []
        while len(todo) > 1:
            expr = todo.pop()
            if isinstance(expr, Plus):
                todo.extend(expr.children)
            else:
                ret.append(expr)
        return tuple(ret)

    def input_form(self) -> str:
        return "(" + " + ".join(child.input_form() for child in self.children) + ")"

    def _collect_terms(self) -> list[Expr]:
        todo = list(self.children)
        ret: list[Expr] = []
        while todo:
            child = todo.pop()
            if isinstance(child, Plus):
                todo.extend(child.children)
            else:
                ret.append(child)
        return ret

    @staticmethod
    def _separate_constant_terms(terms: list[Expr]) -> tuple[Constant, list[Expr]]:
        constant_term = Int(0)
        other_terms: list[Expr] = []
        for term in terms:
            if isinstance(term, Constant):
                constant_term = constant_term.const_add(term)
            else:
                other_terms.append(term)
        return constant_term.normalize(), other_terms

    def evaluate(self, context: 'Context | None' = None) -> 'Expr':
        constant_term, other_terms = self._separate_constant_terms([expr.evaluate(context) for expr in self._collect_terms()])

        term_to_factor: dict[Expr, Constant] = {}
        for term in other_terms:
            cf, factor = term.separate_constant_coefficient()
            term_to_factor[factor] = term_to_factor.get(factor, Int(0)).const_add(cf)

        new_terms: list[Expr] = []
        for factor, cf in term_to_factor.items():
            cf = cf.normalize()
            if cf.is_one():
                new_terms.append(factor)
            elif not cf.is_zero():
                new_terms.append(Times((cf, factor)).evaluate(context))

        if not constant_term.is_zero():
            new_terms.insert(0, constant_term)
        new_terms.sort(key=lambda x: x)

        new_terms2 = tuple(new_terms)

        return Plus.make(new_terms2) if new_terms2 != self.children else self

def peer_type(types: tuple[Expr, ...]) -> Expr:
    assert len(types) > 0
    if len(types) == 1:
        return types[0]
    ret = types[0]
    for t in types[1:]:
        if not ret.is_subtype(t):
            assert t.is_subtype(ret), f"incompatible types {t} and {ret}"
            ret = t
    return ret

@exprclass
class Times(Expr):
    children: tuple[Expr, ...]
    type: Expr = field(init=False, compare=False)
    shape: tuple[Expr, ...] | None = field(init=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "type", peer_type(tuple(i.get_type() for i in self.children)))
        object.__setattr__(self, "shape", merge_shapes(tuple(i.get_shape() for i in self.children)))

    @override
    def get_type(self) -> Expr:
        return self.type

    @staticmethod
    def make(children: tuple[Expr, ...]) -> Expr:
        if len(children) == 0:
            return Int(1)
        if len(children) == 1:
            return children[0]
        return Times(children)

    @staticmethod
    def connect_factors(*start) -> list[Expr]:
        todo = list(start)
        ret: list[Expr] = []
        while len(todo) > 1:
            expr = todo.pop()
            if isinstance(expr, Times):
                todo.extend(expr.children)
            else:
                ret.append(expr)
        return ret

    def input_form(self) -> str:
        return "(" + " * ".join(child.input_form() for child in self.children) + ")"

    @staticmethod
    def _separate_factors_and_power(factors: tuple[Expr, ...]) -> tuple[Constant, tuple[Expr, ...]]:
        constant_factor: Constant = Int(1)
        other_factors: list[Expr] = []
        for child in factors:
            if isinstance(child, Constant):
                constant_factor = constant_factor.const_mul(child)
            else:
                other_factors.append(child)
        return constant_factor.normalize(), tuple(other_factors)

    def _collect_factors(self) -> list[Expr]:
        todo = list(self.children)
        ret: list[Expr] = []
        while todo:
            child = todo.pop()
            if isinstance(child, Times):
                todo.extend(child.children)
            else:
                ret.append(child)
        return ret

    @override
    def separate_constant_coefficient(self) -> tuple['Constant', 'Expr']:
        constant_factor, other_factors = Times._separate_factors_and_power(self.children)
        return constant_factor, Times.make(other_factors)

    @override
    def evaluate(self, context: 'Context | None' = None) -> 'Expr':
        constant_factor, other_factors = Times._separate_factors_and_power(tuple(child.evaluate(context) for child in self._collect_factors()))

        # simple case
        if constant_factor.is_zero():
            return Int(0)
        for factor in other_factors:
            if factor.is_zero():
                return Int(0)

        base_to_exponents: dict[Expr, list[Expr]] = {}
        for child in other_factors:
            p = child.as_power()
            base_to_exponents.setdefault(p.base, []).append(p.exponent)

        new_factors: list[Expr] = []
        for base, exponents in base_to_exponents.items():
            assert len(exponents) > 0
            new_factor: Expr
            if len(exponents) == 1:
                new_factor = Power.make(base, exponents[0])
            else:
                new_factor = Power(base, Plus(tuple(exponents))).evaluate(context)

            if isinstance(new_factor, Constant):
                constant_factor = constant_factor.const_mul(new_factor)
            else:
                new_factors.append(new_factor)

        constant_factor = constant_factor.normalize()

        if not constant_factor.is_one():
            new_factors.insert(0, constant_factor)
        new_factors.sort(key=lambda x: x)

        new_factors2 = tuple(new_factors)

        return Times.make(new_factors2) if new_factors2 != self.children else self

@exprclass
class Power(Expr):
    base: Expr
    exponent: Expr

    @override
    def get_type(self) -> Expr:
        return self.base.get_type()

    def get_shape(self) -> tuple[Expr, ...] | None:
        return self.base.get_shape()

    @staticmethod
    def make(base: Expr, exponent: Expr) -> Expr:
        if exponent.is_zero():
            return Int(1)
        match exponent:
            case Int(value):
                if value == 1:
                    return base
                if value == 0:
                    return Int(0)
        return Power(base, exponent)

    @override
    def as_power(self) -> 'Power':
        return self

    @override
    def evaluate(self, context: 'Context | None' = None) -> 'Expr':
        base = self.base.evaluate(context)
        exponent = self.exponent.evaluate(context)

        # simple cases
        if base.is_zero() and isinstance(exponent, Int) and exponent.value > 0:
            return Int(0)
        if exponent.is_zero():
            return Int(1)

        if isinstance(exponent, Int):
            if isinstance(base, Times):
                factors = tuple((a ** exponent).evaluate(context) for a in base.children)
                return Times(factors).evaluate(context)
            if isinstance(base, Constant):
                return base.int_pow(exponent.value).normalize()

        return Power(base, exponent) if base is not self.base or exponent is not self.exponent else self

    def input_form(self) -> str:
        return f"({self.base.input_form()} ^ {self.exponent.input_form()})"

@exprclass
class Apply(Expr):
    fn: Expr
    args: list[Expr]

@exprclass
class UnaryNumericFunction(Expr):
    expr: Expr

    @override
    def get_type(self) -> Expr:
        type = self.expr.get_type()
        if isinstance(type, ComplexType):
            return type
        return RealType()

    def get_shape(self) -> tuple[Expr, ...] | None:
        return self.expr.get_shape()

class Sin(UnaryNumericFunction):
    @override
    def input_form(self) -> str:
        return f"sin({self.expr.input_form()})"

class Cos(UnaryNumericFunction):
    @override
    def input_form(self) -> str:
        return f"cos({self.expr.input_form()})"

class Ln(UnaryNumericFunction):
    @override
    def input_form(self) -> str:
        return f"ln({self.expr.input_form()})"

class Exp(UnaryNumericFunction):
    @override
    def input_form(self) -> str:
        return f"exp({self.expr.input_form()})"

@exprclass
class Roll(Expr):
    expr: Expr
    axis: int
    amount: int

    @override
    def get_type(self) -> Expr:
        return self.expr.get_type()

    def get_shape(self) -> tuple[Expr, ...] | None:
        return self.expr.get_shape()

@exprclass
class Slice(Expr):
    expr: Expr
    axis: int
    index: int
    shape: tuple[Expr, ...] | None = field(init=False, compare=False)

    def __post_init__(self) -> None:
        shape = self.expr.get_shape()
        if shape is None:
            object.__setattr__(self, 'shape', None)
            return
        if self.axis >= len(shape):
            raise IndexError(f"Axis {self.axis} is out of bounds for shape {shape}")
        object.__setattr__(self, 'shape', shape[:self.axis] + shape[self.axis + 1:])

    @override
    def get_type(self) -> 'Expr':
        return self.expr.get_type()

    def get_shape(self) -> tuple[Expr, ...] | None:
        return self.shape

    @override
    def input_form(self) -> str:
        return f"({self.expr.input_form()} [{self.axis}, {self.index}])"

class AssignExpr:
    lhs: Expr
    rhs: Expr
    type: Expr
    shape: tuple[Expr, ...] | None
    op: str

    def __init__(self, lhs: Expr, rhs: Expr, op: str = '') -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert lhs_type.is_subtype(rhs_type), f"cannot assign type {rhs_type} to {lhs_type}"
        self.type = lhs_type

        lhs_shape = lhs.get_shape()
        rhs_shape = rhs.get_shape()
        shape = merge_shape(lhs_shape, rhs_shape)
        assert shape == lhs_shape, "incompatible shape"
        self.shape = shape

    def __str__(self) -> str:
        return f"{self.lhs} {self.op}= {self.rhs}"

    def total_size(self):
        if self.shape is not None:
            return Times.make(self.shape).evaluate()
        else:
            return None

def derivative(expr: Expr, var: Expr, default_case_handler: Callable[[Expr, Expr], Expr] | None = None) -> Expr:
    match expr:
        case Constant():
            return Int(0)
        case Plus(children):
            return Plus(tuple(derivative(child, var, default_case_handler) for child in children))
        case Times(children):
            terms: list[Expr] = []
            for i in range(len(children)):
                factors: list[Expr] = []
                for j in range(len(children)):
                    if i != j:
                        factors.append(children[j])
                    else:
                        factors.append(derivative(children[j], var, default_case_handler))
                terms.append(Times(tuple(factors)))
            return Plus(tuple(terms))
        case Power(base, exponent):
            return exponent * base ** (exponent - 1) * derivative(base, var, default_case_handler) + Ln(base) * base ** exponent * derivative(exponent, var, default_case_handler)
        case Sin(expr):
            return Cos(expr) * derivative(expr, var, default_case_handler)
        case Cos(expr):
            return Times((Int(-1), Sin(expr))) * derivative(expr, var, default_case_handler)
        case Ln(expr):
            return derivative(expr, var, default_case_handler) / expr
        case Exp(expr):
            return Exp(expr) * derivative(expr, var, default_case_handler)
        case Roll() | Slice():
            raise ValueError(f"Cannot take derivative of {expr}")
        case _:
            if default_case_handler is not None:
                return default_case_handler(expr, var)
            return Int(1) if expr == var else Int(0)
