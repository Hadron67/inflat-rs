"""The spy type system of the compile-time interpreter.

Types appear in two roles:

* as the type annotation values in AOT functions (``spy.u64``,
  ``spy.f64``, ...), and
* as compile-time values inside a function body (``spy.type(a) ==
  spy.u64``).

The static types attached to the registers of the typed MIR are the
mirrors of these types defined by ``mir``; the interpreter converts
between the two when it emits instructions.

Types are immutable and compare structurally (two ``IntType(64, False)``
are equal), which is what makes the compile-time comparisons in
``spy.type(a) == spy.u64`` work.
"""

from abc import abstractmethod
from dataclasses import dataclass
from types import FunctionType as PyFunctionType
from typing import Any, override

from . import mir

INT_DEFAULT_BITS = 32
"""A plain Python ``int`` argument is mapped to this signedness/width by
default (see ``value_type``)."""


class Value:
    """Base of the *spy values* of the compile-time domain: types
    (used as values by ``spy.type``) and other compile-time objects.
    Concrete values expose their spy type as ``.type``."""
    @abstractmethod
    def type(self) -> 'Type':
        raise NotImplementedError

class Type(Value):
    @override
    def type(self) -> 'Type':
        return TYPE_TYPE

@dataclass(frozen=True)
class TypeType(Type):
    level: int
    @override
    def type(self) -> Type:
        return TypeType(self.level + 1)

TYPE_TYPE = TypeType(0)

@dataclass(frozen=True)
class BoolType(Type):
    """The boolean type; values are ``i1`` at the LLVM level."""

    @override
    def type(self) -> Type:
        return TYPE_TYPE

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
class FunctionType(Type):
    args: tuple[FormalArg, ...]
    return_type: Type


# ---------------------------------------------------------------------------
# function values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnyFunction(Type):
    """The type of a function value whose signature is not known: a lazy
    ``@jit`` function is only typed when a call specializes it.  It has
    no MIR mirror - such a value never crosses into runtime code."""


class SpyFunction(Value):
    """A function value: the compile-time value standing for a spy
    function registered in a :class:`JitContext`.  Function values are
    identity objects: two are equal only if they are the same object.

    A value doubles as the per-function entry of its host context: it
    holds the Python function, the function kind, the context-unique
    base of its native symbol names and the compiled specializations.
    The call logic itself lives in the interpreter and the host, not
    here.
    """

    def __init__(self, fn: PyFunctionType, kind: str) -> None:
        self.fn = fn
        self.kind = kind  # 'jit' or 'aot'
        # the hosting JitContext (set when the value is registered),
        # only used when checking whether function is called within the same context
        self.context: Any = None
        # the context-unique base name of the native symbols
        self.name_base = ''
        # spy argument types -> compiled native function (see ``lower``)
        self.specs: dict[tuple[Type, ...], Any] = {}
        self.failed: dict[tuple[Type, ...], str] = {}

    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)


class LazyJitFunction(SpyFunction):
    """The function value of a ``@jit`` function: it is only compiled
    (and thereby typed) when a call specializes it, so as a value its
    type is the untyped :class:`AnyFunction`."""

    def __init__(self, fn: PyFunctionType) -> None:
        super().__init__(fn, 'jit')

    @override
    def type(self) -> Type:
        return AnyFunction()


class FunctionValue(SpyFunction):
    """The function value of a ``@aot`` function: compiled from its
    type annotations when it is registered, so the value carries the
    concrete signature and the compiled :class:`mir.Function` (calling
    it emits a ``mir.Call`` of that function)."""

    def __init__(
        self,
        fn: PyFunctionType,
        args: tuple[FormalArg, ...],
        ret: Type,
        mir_fn: mir.Function | None = None,
    ) -> None:
        super().__init__(fn, 'aot')
        self.args = args
        self.ret = ret
        self.mir_fn = mir_fn

    @override
    def type(self) -> Type:
        return PointerType(FunctionType(self.args, self.ret), True)

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
            return '*' + ('const ' if is_const else '') + type_str(elem)
        case FunctionType(args, ret):
            return f'fn({', '.join(type_str(a.type) for a in args)}) -> {type_str(ret)}'
        case AnyFunction():
            return 'any fn'
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
