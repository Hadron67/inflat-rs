"""The typed MIR.

The interpreter ("runs" the untyped HIR) emits a :class:`Function`: a
linear list of typed instructions plus the concrete types of the formal
arguments and of the return value.  Executing the function body emits
straight-line code only; runtime control flow (runtime ``if``/``while``)
will be added later as explicit basic blocks with branches and phis.

The representation is deliberately close to LLVM so that ``lower`` is a
mechanical mapping; every MIR value exposes a ``.type`` (a
``type.Type``).
"""

from dataclasses import dataclass

from .type import BoolType, FormalArg, PointerType, Type, Value


@dataclass
class Param(Value):
    """The index-th by-value argument of the enclosing function."""

    index: int
    type: Type
    name: str = ''


class Inst(Value):
    """A MIR instruction; the object itself acts as its result register
    (instructions have identity, mirroring ``symlat.jit.llvm``)."""

    def __eq__(self, other: object, /) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(eq=False)
class Alloca(Inst):
    """Allocate a slot for one value; produces a pointer to ``type``."""

    type: PointerType


@dataclass(eq=False)
class Load(Inst):
    ptr: Value
    type: Type


@dataclass(eq=False)
class Store(Inst):
    ptr: Value
    value: Value


@dataclass(eq=False)
class Arith(Inst):
    """Integer/float arithmetic.

    ``op`` is one of ``'add'``, ``'sub'``, ``'mul'`` (integer or float,
    chosen by ``type``), ``'div'`` and ``'rem'`` (float division is
    ``'div'`` with a float result type).  Integer division/remainder
    honor ``signed``.
    """

    op: str
    signed: bool
    lhs: Value
    rhs: Value
    type: Type


@dataclass(eq=False)
class Convert(Inst):
    """A value conversion: 'sitofp', 'uitofp', 'fpext', 'fptrunc',
    'zext', 'sext', 'trunc', 'fptosi' or 'fptoui'."""

    kind: str
    value: Value
    type: Type


@dataclass(eq=False)
class Cmp(Inst):
    """A comparison producing a bool; ``op`` is one of 'eq', 'ne', 'lt',
    'le', 'gt', 'ge'."""

    op: str
    signed: bool
    kind: str  # 'int' or 'float'
    lhs: Value
    rhs: Value

    @property
    def type(self) -> Type:
        return BoolType()


@dataclass(eq=False)
class Call(Inst):
    """A call to a native function previously compiled by the same
    JitContext (an LLVM ``call`` to a declared symbol)."""

    callee_name: str
    args: tuple[Value, ...]
    type: Type


@dataclass(eq=False)
class Ret(Inst):
    value: Value


@dataclass
class Function:
    name: str
    args: tuple[FormalArg, ...]
    ret_type: Type
    insts: list[Inst]


def type_of(value: Value) -> Type:
    return value.type  # type: ignore[attr-defined]
