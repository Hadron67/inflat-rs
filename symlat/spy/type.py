"""The type system of spy functions.

Types appear in three roles:

* as the type annotation values in AOT functions (``spy.u64``,
  ``spy.f64``, ...),
* as compile-time values inside a function body (``spy.type(a) ==
  spy.u64``), and
* as the static types attached to the values of the typed MIR.

Types are immutable and compare structurally (two ``IntType(64, False)``
are equal), which is what makes the compile-time comparisons in
``spy.type(a) == spy.u64`` work.
"""

from dataclasses import dataclass

INT_DEFAULT_BITS = 32
"""A plain Python ``int`` argument is mapped to this signedness/width by
default (see ``value_type``)."""


class Type:
    pass


class Value:
    """Base of all *spy values*: typed constants (in this module) and the
    typed MIR registers of ``mir``.  Concrete values expose their spy
    type as ``.type``."""


class BoolType(Type):
    """The boolean type; values are ``i1`` at the LLVM level."""

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, BoolType)

    def __hash__(self) -> int:
        return hash('spy.bool')

    def __repr__(self) -> str:
        return 'bool'


@dataclass(frozen=True)
class IntType(Type):
    bits: int
    signed: bool


@dataclass(frozen=True)
class FloatType(Type):
    bits: int

    def __post_init__(self) -> None:
        assert self.bits in (32, 64), f"unsupported float bits {self.bits}"


@dataclass(frozen=True)
class PointerType(Type):
    elem: Type
    is_const: bool = False


@dataclass(frozen=True)
class FormalArg:
    name: str
    type: Type


@dataclass(frozen=True)
class FunctionPointerType(Type):
    args: tuple[FormalArg, ...]
    return_type: Type


# ---------------------------------------------------------------------------
# typed constants (MIR operands)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntValue(Value):
    value: int
    bits: int
    signed: bool

    @property
    def type(self) -> Type:
        return IntType(self.bits, self.signed)


@dataclass(frozen=True)
class FloatValue(Value):
    value: float
    bits: int

    @property
    def type(self) -> Type:
        return FloatType(self.bits)


@dataclass(frozen=True)
class BoolValue(Value):
    value: bool

    @property
    def type(self) -> Type:
        return BoolType()


def int_range(type: IntType) -> tuple[int, int]:
    if type.signed:
        return (-(2 ** (type.bits - 1)), 2 ** (type.bits - 1) - 1)
    return (0, 2 ** type.bits - 1)


def type_str(type: Type) -> str:
    """A short, printable name of a type (used in error messages and in the
    mangled names of compiled specializations)."""
    match type:
        case BoolType():
            return 'bool'
        case IntType():
            return ('i' if type.signed else 'u') + str(type.bits)
        case FloatType():
            return 'f' + str(type.bits)
        case PointerType(elem, is_const):
            return ('const ' if is_const else '') + type_str(elem) + '*'
        case FunctionPointerType(args, ret):
            return f'fn({', '.join(type_str(a.type) for a in args)}) -> {type_str(ret)}'
        case _:
            return str(type)


# ---------------------------------------------------------------------------
# mapping Python values to spy types
# ---------------------------------------------------------------------------


def value_type(value: object) -> Type | None:
    """The spy type a Python *value* is marshaled to at the call boundary.

    ``None`` is returned for values that have no spy representation (e.g.
    compile-time objects like type descriptors, which never cross the
    boundary).
    """
    match value:
        case bool():
            return BoolType()
        case int():
            return IntType(INT_DEFAULT_BITS, True)
        case float():
            return FloatType(64)
        case str():
            # strings are compiled as arrays of u8; until arrays get their
            # own type they are represented by a const pointer to u8
            return PointerType(IntType(8, False), is_const=True)
        case _:
            return None


def value_repr(value: object) -> str:
    """Human readable description of a Python value (used in errors)."""
    match value:
        case int():
            return f'integer {value}'
        case float():
            return f'float {value}'
        case str():
            return 'string'
        case bool():
            return 'bool'
        case _:
            return repr(value)
