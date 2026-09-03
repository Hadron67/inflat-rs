"""The user-facing DSL: ``JitContext`` with the ``jit`` and ``aot``
decorators.

A decorated function becomes a callable wrapper.  At call time the
wrapper

1. binds the Python arguments to the formal parameters (keyword
   arguments and default values are filled in here),
2. solves the concrete spy types of the parameters (in jit mode from
   the marshaled types of the arguments plus type-parameter unification,
   in aot mode from the annotations),
3. marshals every argument to the parameter types,
4. makes sure the specialization for those types is compiled (the
   compile pipeline is ``astgen -> hir -> interp (typed mir) -> lower``)
   and calls the native function.

Calling a *decorated* function from inside another spy function goes
through the same resolution but is compiled to a native ``call``; see
``interp``.
"""

from types import FunctionType
from typing import Any

from . import astgen
from .builtins import AsValue
from .errors import CompileError, SpyError, TypeMismatchError
from .interp import CallTarget, FunctionResolver, HirRunner
from .lower import NativeFn, compile_native
from .type import (
    BoolType,
    FloatType,
    IntType,
    PointerType,
    Type,
    int_range,
    type_str,
    value_type,
)


def _sanitize(name: str) -> str:
    return ''.join(c if (c.isalnum() or c in '_.') else '_' for c in name)


def symbol_of(fn_name: str, arg_types: tuple[Type, ...]) -> str:
    """The native symbol name of one specialization.  ``fn_name`` must
    be the context-unique name allocated at registration (see
    ``JitContext._allocate_name``); the suffix of ``type_str`` strings
    distinguishes the specializations of one function."""
    parts = ['spy', _sanitize(fn_name)]
    parts.extend(_sanitize(type_str(t)) for t in arg_types)
    return '.'.join(parts)


def _types_str(arg_types: tuple[Type, ...]) -> str:
    return ', '.join(type_str(t) for t in arg_types)


def _candidate_type(fn_name: str, param_name: str, value: object) -> Type:
    """The spy type a provided Python argument marshals to."""
    if isinstance(value, AsValue):
        return value.type
    t = value_type(value)
    if t is None:
        raise TypeMismatchError(
            f"cannot pass a {type(value).__name__} value as the '{param_name}' argument "
            f"of {fn_name} (not a spy value)"
        )
    return t


def _marshal(fn_name: str, param_name: str, value: object, target: Type) -> object:
    """Convert one Python argument to the native calling convention of
    ``target`` (returns the value handed to the ctypes entry point)."""
    if isinstance(value, AsValue):
        if value.type != target:
            raise TypeMismatchError(
                f"type mismatch: spy.as(..., {type_str(value.type)}) cannot be passed "
                f"as the '{param_name}' argument of {fn_name} (expected "
                f"{type_str(target)})"
            )
        value = value.value

    def fail() -> TypeMismatchError:
        kind = 'a bool' if isinstance(value, bool) else f'a {type(value).__name__}'
        return TypeMismatchError(
            f"type mismatch: {kind} value cannot be passed as the '{param_name}' "
            f"argument of {fn_name} (expected {type_str(target)})"
        )

    match target:
        case BoolType():
            if not isinstance(value, bool):
                raise fail()
            return bool(value)
        case IntType():
            if isinstance(value, bool) or not isinstance(value, int):
                raise fail()
            lo, hi = int_range(target)
            if not lo <= value <= hi:
                raise TypeMismatchError(
                    f"integer {value} is out of range for type {type_str(target)} "
                    f"in argument '{param_name}' of {fn_name}"
                )
            return int(value)
        case FloatType():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise fail()
            try:
                return float(value)
            except OverflowError:
                raise TypeMismatchError(
                    f"cannot convert {value} to {type_str(target)} in argument "
                    f"'{param_name}' of {fn_name}"
                ) from None
        case PointerType():
            if not isinstance(value, str):
                raise fail()
            return value.encode('utf-8')
        case _:
            raise TypeMismatchError(
                f"type {type_str(target)} is not supported for arguments of {fn_name} yet"
            )


class FnEntry:
    """One decorated function of one :class:`JitContext`."""

    def __init__(self, context: 'JitContext', fn: FunctionType, kind: str) -> None:
        self.context = context
        self.fn = fn
        self.kind = kind  # 'jit' or 'aot'
        # the context-unique base of the native symbol names of this
        # function's specializations (allocated by ``JitContext``)
        self.name_base = ''
        self.specs: dict[tuple[Type, ...], NativeFn] = {}
        self.failed: dict[tuple[Type, ...], str] = {}

    def dispatch(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        fn_ir = self.context.hir_of(self.fn)
        name = self.fn.__name__
        params = fn_ir.params
        param_names = [p.name for p in params]

        if len(args) > len(params):
            raise TypeError(
                f"{name}() takes {len(params)} positional arguments but {len(args)} were given"
            )
        present: dict[str, Any] = {}
        for i, value in enumerate(args):
            present[param_names[i]] = value
        for key, value in kwargs.items():
            if key not in param_names:
                raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
            if key in present:
                raise TypeError(f"{name}() got multiple values for argument '{key}'")
            present[key] = value

        provided: list[Type | None] = []
        for param in params:
            if param.name in present:
                provided.append(_candidate_type(name, param.name, present[param.name]))
            else:
                provided.append(None)
        formal = astgen.solve_call_types(fn_ir, self.kind, tuple(provided))

        marshaled: list[Any] = []
        for i, param in enumerate(params):
            if param.name in present:
                value = present[param.name]
            else:
                if not param.has_default:
                    raise TypeError(f"{name}() missing required argument '{param.name}'")
                value = param.default_value
            marshaled.append(_marshal(name, param.name, value, formal[i]))

        spec = self.context.ensure_spec(self, formal)
        return spec.call(*marshaled)


class _SpyFn:
    __slots__ = ('_spy_entry',)

    def __init__(self, entry: FnEntry) -> None:
        self._spy_entry = entry

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._spy_entry.dispatch(args, kwargs)

    @property
    def __name__(self) -> str:
        return self._spy_entry.fn.__name__

    def __repr__(self) -> str:
        kind = self._spy_entry.kind
        return f'<spy {kind} function {self._spy_entry.fn.__name__}>'


class JitContext(FunctionResolver):
    """A cache of compiled spy functions; functions decorated by the
    same context may call each other (as native calls)."""

    def __init__(self) -> None:
        self._entries: dict[FunctionType, FnEntry] = {}
        self._hir_cache: dict[FunctionType, astgen.FunctionIR] = {}
        self._symbols: dict[str, NativeFn] = {}
        self._compiling: set[tuple[object, tuple[Type, ...]]] = set()
        # allocated base names (``spy.<name>.<types>``) -> their owner;
        # keeps the native symbols of different functions apart even when
        # they happen to share a ``__name__``
        self._name_owners: dict[str, FnEntry] = {}

    # -- decorators ----------------------------------------------------------

    def jit(self, fn: FunctionType | None = None):
        """``@cache.jit()``: compile lazily at the first call, using the
        marshaled types of the arguments (annotations have no effect in
        jit mode, except for unifying type parameters like ``T``)."""
        if fn is not None:
            return self._register(fn, 'jit')
        return lambda f: self._register(f, 'jit')

    def aot(self, fn: FunctionType | None = None):
        """``@cache.aot()``: compile immediately from the (concrete)
        type annotations, which are required for every parameter and for
        the return type."""
        if fn is not None:
            return self._register(fn, 'aot')
        return lambda f: self._register(f, 'aot')

    # -- registry ------------------------------------------------------------

    def hir_of(self, fn: FunctionType) -> astgen.FunctionIR:
        ir = self._hir_cache.get(fn)
        if ir is None:
            ir = astgen.parse_function(fn)
            self._hir_cache[fn] = ir
        return ir

    def ensure_spec(self, entry: FnEntry, arg_types: tuple[Type, ...]) -> NativeFn:
        """Compile (or look up) the specialization of ``entry`` for
        ``arg_types`` and return its native function."""
        spec = entry.specs.get(arg_types)
        if spec is not None:
            return spec
        message = entry.failed.get(arg_types)
        if message is not None:
            raise CompileError(message)

        fn_name = entry.fn.__name__
        key = (entry, arg_types)
        if key in self._compiling:
            raise CompileError(
                f"recursive calls are not supported yet: {fn_name}({_types_str(arg_types)})"
            )
        try:
            self._compiling.add(key)
            fn_ir = self.hir_of(entry.fn)
            ret_hint: Type | None = None
            if entry.kind == 'aot':
                ret_hint = astgen.return_annotation_type(fn_ir)
            runner = HirRunner(self)
            mir_fn, _ = runner.run_function(
                fn_ir, symbol_of(entry.name_base, arg_types), arg_types, ret_hint
            )
            native = compile_native(mir_fn.name, mir_fn, self._extern_symbols())
            entry.specs[arg_types] = native
            self._symbols[native.name] = native
            return native
        except SpyError as e:
            entry.failed[arg_types] = str(e)
            raise CompileError(
                f"error while compiling {fn_name}({_types_str(arg_types)}): {e}"
            ) from e
        finally:
            self._compiling.discard(key)

    def resolve_call(self, entry: FnEntry, arg_types: tuple[Type, ...]) -> CallTarget:
        """Resolve a native call of one specialization from inside a
        compiled function (may trigger a nested compilation)."""
        native = self.ensure_spec(entry, arg_types)
        return CallTarget(native.name, native.ret_type)

    # -- internals -----------------------------------------------------------

    def _register(self, fn: FunctionType, kind: str) -> _SpyFn:
        if not isinstance(fn, FunctionType):
            raise TypeError('spy decorators can only be applied to plain Python functions')
        if fn in self._entries:
            raise ValueError(f'function {fn.__name__} is already registered in this JitContext')
        entry = FnEntry(self, fn, kind)
        entry.name_base = self._allocate_name(fn, entry)
        self._entries[fn] = entry
        wrapper = _SpyFn(entry)
        if kind == 'aot':
            # compile immediately from the annotations
            fn_ir = self.hir_of(fn)
            formal = astgen.solve_call_types(
                fn_ir, 'aot', (None,) * len(fn_ir.params)
            )
            self.ensure_spec(entry, formal)
        return wrapper

    def _allocate_name(self, fn: FunctionType, entry: FnEntry) -> str:
        """The context-unique base name of the native symbols of ``fn``.

        Two different functions registered in the same context may share
        a ``__name__`` (e.g. same-named functions of two modules, or two
        instances of a nested function); their symbols must still not
        collide, because extern calls are linked by name.  The plain
        ``__name__`` is kept when it is free; otherwise the
        module-qualified name is used, then numbered suffixes."""
        candidates: list[str] = [_sanitize(fn.__name__)]
        qualified = _sanitize(f'{fn.__module__}.{fn.__qualname__}')
        if qualified not in candidates:
            candidates.append(qualified)
        name: str | None = None
        for candidate in candidates:
            if candidate not in self._name_owners:
                name = candidate
                break
        if name is None:
            n = 2
            while True:
                candidate = f'{qualified}.{n}'
                if candidate not in self._name_owners:
                    name = candidate
                    break
                n += 1
        assert name is not None
        self._name_owners[name] = entry
        return name

    def _extern_symbols(self) -> dict[str, int]:
        return {name: native.addr for name, native in self._symbols.items()}
