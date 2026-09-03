
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FunctionType as PyFunctionType
from typing import Any, TypeVar, override

from . import hir, mir
from .type import AnyFunction, FormalArg, FunctionType, PointerType, Type, Value


@dataclass(frozen=True)
class ParamDef:
    name: str
    # The evaluated annotation from ``fn.__annotations__``: either a concrete
    # spy type or one of the function's PEP 695 type parameter objects.
    annotation: Any | None = None
    has_default: bool = False
    default_value: Any | None = None


@dataclass
class FunctionIR:
    fn: Callable
    name: str
    # The declared PEP 695 type parameter objects (``[T]``), kept so that
    # annotation values can be recognized as type parameters by identity.
    type_params: tuple[TypeVar, ...]
    params: tuple[ParamDef, ...]
    # The evaluated return annotation from ``fn.__annotations__``.
    ret_annotation: Any | None
    body: tuple[hir.Inst, ...]

@dataclass
class NativeFn:
    """A compiled native function of one specialization."""

    name: str
    arg_types: tuple[mir.Type, ...]
    ret_type: mir.Type
    lines: list[str] = field(default_factory=list)
    _engine: object = None  # type: ignore[assignment]
    _addr: int = 0
    _entry: Any = None

    def call(self, *values) -> object:
        return self._entry(*values)

    @property
    def addr(self) -> int:
        return self._addr

    def print_all(self) -> list[str]:
        return self.lines

class SpyFunction(Value):
    """A function value: the compile-time value standing for a spy
    function registered in a :class:`JitContext`.  Function values are
    identity objects: two are equal only if they are the same object.

    A value doubles as the per-function entry of its host context: it
    holds the Python function, the function kind, the context-unique
    base of its native symbol names, the parsed HIR, the typed MIR of
    every compiled specialization and the compiled native functions.
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
        # the parsed HIR of the function (see ``JitContext.hir_of``)
        self.hir: FunctionIR | None = None
        # spy argument types -> the typed MIR function of the
        # specialization.  The function is registered here - with an
        # empty body - before its body is typed, so a recursive call
        # made by the body resolves to it.  A function whose module
        # build aborted stays cached until the next build that
        # references it (its spec is only registered when the module it
        # was lowered into finishes).
        self.mir_cache: dict[tuple[Type, ...], mir.Function] = {}
        # spy argument types -> compiled native function (see ``lower``)
        self.specs: dict[tuple[Type, ...], NativeFn] = {}
        # spy argument types -> error message of a failed compilation
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
