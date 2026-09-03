"""The user-facing DSL: ``JitContext`` with the ``jit`` and ``aot``
decorators.

A decorated function becomes a *function value* (``LazyJitFunction``
for ``jit``, ``FunctionValue`` for ``aot``), mounted as ``_spy_entry``
on the function object; Python-side calls go through a thin callable
view bound to its context.  At call time the view

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

from symlat.spy import mir

from . import astgen
from .builtins import AsValue
from .errors import CompileError, SpyError, TypeMismatchError
from .interp import FunctionResolver, HirRunner
from .lower import NativeFn, compile_module
from .mir import Function as MirFunction
from .mir import FunctionType as MirFunctionType
from .mir import Symbol as MirSymbol
from .type import (
    BoolType,
    FloatType,
    FormalArg,
    FunctionValue,
    IntType,
    LazyJitFunction,
    PointerType,
    SpyFunction,
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


class _FunctionView:
    """A callable view of a function value.  Function values themselves
    are pure values (the call logic lives in the interpreter and the
    host), so Python-side calls go through this thin view bound to its
    context.  The view exposes its value as ``_spy_entry`` (also
    mounted on the raw function, so name resolution can find it)."""

    __slots__ = ('_context', '_spy_entry')

    def __init__(self, context: 'JitContext', entry: SpyFunction) -> None:
        self._context = context
        self._spy_entry = entry

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._context._dispatch(self._spy_entry, args, kwargs)

    @property
    def __name__(self) -> str:
        return self._spy_entry.fn.__name__

    def __repr__(self) -> str:
        entry = self._spy_entry
        return f'<spy {entry.kind} function {entry.fn.__name__}>'


class JitContext(FunctionResolver):
    """A cache of compiled spy functions; functions decorated by the
    same context may call each other (as native calls)."""

    def __init__(self) -> None:
        self._entries: dict[FunctionType, SpyFunction] = {}
        self._hir_cache: dict[FunctionType, astgen.FunctionIR] = {}
        self._symbols: dict[str, NativeFn] = {}
        self._compiling: set[tuple[object, tuple[Type, ...]]] = set()
        # allocated base names (``spy.<name>.<types>``) -> their owner;
        # keeps the native symbols of different functions apart even when
        # they happen to share a ``__name__``
        self._name_owners: dict[str, SpyFunction] = {}
        # typed MIR of every compiled specialization, keyed like
        # ``_compiling``; survives the module it was compiled into
        self._mir_cache: dict[tuple[SpyFunction, tuple[Type, ...]], MirFunction] = {}
        # the MIR functions being compiled into the module under
        # construction (see ``ensure_spec``); None outside a build
        self._module: dict[tuple[SpyFunction, tuple[Type, ...]], MirFunction] | None = None

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

    # -- Python-side calls ---------------------------------------------------

    def _dispatch(self, entry: SpyFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Bind Python arguments to the formal parameters of ``entry``
        and call the (possibly just compiled) specialization; this is
        what the callable view of a function value forwards to."""
        fn_ir = self.hir_of(entry.fn)
        name = entry.fn.__name__
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
        formal = astgen.solve_call_types(fn_ir, entry.kind, tuple(provided))

        marshaled: list[Any] = []
        for i, param in enumerate(params):
            if param.name in present:
                value = present[param.name]
            else:
                if not param.has_default:
                    raise TypeError(f"{name}() missing required argument '{param.name}'")
                value = param.default_value
            marshaled.append(_marshal(name, param.name, value, formal[i]))

        spec = self.ensure_spec(entry, formal)
        return spec.call(*marshaled)

    # -- registry ------------------------------------------------------------

    def hir_of(self, fn: FunctionType) -> astgen.FunctionIR:
        ir = self._hir_cache.get(fn)
        if ir is None:
            ir = astgen.parse_function(fn)
            self._hir_cache[fn] = ir
        return ir

    def ensure_spec(self, entry: SpyFunction, arg_types: tuple[Type, ...]) -> NativeFn:
        """Compile (or look up) the specialization of ``entry`` for
        ``arg_types`` and return its native function.

        Compilation happens module-at-a-time: a fresh function and every
        function it depends on that is not compiled yet are lowered
        together into one LLVM module (``define``s); specializations
        compiled in earlier modules are referenced as external symbols.
        """
        spec = entry.specs.get(arg_types)
        if spec is not None:
            return spec
        message = entry.failed.get(arg_types)
        if message is not None:
            raise CompileError(message)
        if self._module is not None:
            raise CompileError('internal error: nested module compilation')

        self._module = {}
        try:
            self._compile_mir(entry, arg_types)
            natives = self._compile_module()
            if isinstance(entry, FunctionValue):
                entry.mir_fn = self._mir_cache[(entry, arg_types)]
            return natives[(entry, arg_types)]
        except SpyError as e:
            entry.failed[arg_types] = str(e)
            raise CompileError(
                f"error while compiling {entry.fn.__name__}({_types_str(arg_types)}): {e}"
            ) from e
        finally:
            self._module = None

    def _compile_mir(self, entry: SpyFunction, arg_types: tuple[Type, ...]) -> MirFunction:
        """Produce (and cache) the typed MIR of one specialization.

        Producing the MIR of a function runs its body, which resolves the
        calls it makes; those may compile (MIR-wise) further functions,
        which are then scheduled into the module under construction (see
        :meth:`ensure_spec`).  The whole module is lowered at once.
        """
        key = (entry, arg_types)
        cached = self._mir_cache.get(key)
        if cached is not None:
            return cached
        if key in self._compiling:
            raise CompileError(
                f"recursive calls are not supported yet: "
                f"{entry.fn.__name__}({_types_str(arg_types)})"
            )
        self._compiling.add(key)
        try:
            fn_ir = self.hir_of(entry.fn)
            ret_hint: Type | None = None
            if entry.kind == 'aot':
                ret_hint = astgen.return_annotation_type(fn_ir)
            runner = HirRunner(self)
            mir_fn, _ = runner.run_function(
                fn_ir, symbol_of(entry.name_base, arg_types), arg_types, ret_hint
            )
            self._mir_cache[key] = mir_fn
            module = self._module
            if module is None:
                raise CompileError(
                    f"internal error: {entry.fn.__name__}({_types_str(arg_types)}) "
                    'is compiled outside of a module build'
                )
            module[key] = mir_fn
            return mir_fn
        finally:
            self._compiling.discard(key)

    def _compile_module(self) -> dict[tuple[SpyFunction, tuple[Type, ...]], NativeFn]:
        """Lower the module under construction (see :meth:`ensure_spec`)
        into one native module and register its specializations."""
        assert self._module is not None
        natives = compile_module(list(self._module.values()), self._extern_symbols())
        by_name = {native.name: native for native in natives}
        result: dict[tuple[SpyFunction, tuple[Type, ...]], NativeFn] = {}
        for key, mir_fn in self._module.items():
            native = by_name[mir_fn.name]
            entry, arg_types = key
            entry.specs[arg_types] = native
            self._symbols[native.name] = native
            result[key] = native
        return result

    def resolve_call(self, entry: SpyFunction, arg_types: tuple[Type, ...]) -> tuple[mir.Value, mir.Type]:
        """The callable value of one callee specialization as seen from
        inside a compiled function: a :class:`mir.Function` when the
        callee is still fresh (it will be ``define``d in the module of
        the caller) or a :class:`mir.Symbol` of a specialization
        compiled in an earlier module."""
        spec = entry.specs.get(arg_types)
        if spec is not None:
            fn_type = MirFunctionType(spec.arg_types, spec.ret_type)
            return MirSymbol(spec.name, fn_type), spec.ret_type
        mir_fn = self._compile_mir(entry, arg_types)
        return mir_fn, mir_fn.ret_type

    # -- internals -----------------------------------------------------------

    def _register(self, fn: FunctionType, kind: str) -> _FunctionView:
        if not isinstance(fn, FunctionType):
            raise TypeError('spy decorators can only be applied to plain Python functions')
        if fn in self._entries:
            raise ValueError(f'function {fn.__name__} is already registered in this JitContext')
        entry: SpyFunction
        formal: tuple[Type, ...] | None = None
        if kind == 'aot':
            # resolve the fixed signature from the annotations first; the
            # body is compiled right below (registration time)
            fn_ir = self.hir_of(fn)
            params = fn_ir.params
            formal = astgen.solve_call_types(fn_ir, 'aot', (None,) * len(params))
            ret = astgen.return_annotation_type(fn_ir)
            args = tuple(FormalArg(params[i].name, formal[i]) for i in range(len(params)))
            entry = FunctionValue(fn, args, ret)
        else:
            entry = LazyJitFunction(fn)
        entry.context = self
        entry.name_base = self._allocate_name(fn, entry)
        self._entries[fn] = entry
        # mount the value on the function object, so that name resolution
        # inside spy bodies finds it (see ``astgen``)
        fn._spy_entry = entry  # type: ignore[attr-defined]
        view = _FunctionView(self, entry)
        if kind == 'aot':
            assert formal is not None
            self.ensure_spec(entry, formal)
        return view

    def _allocate_name(self, fn: FunctionType, entry: SpyFunction) -> str:
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
