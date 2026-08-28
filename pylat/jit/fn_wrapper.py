"""User-facing JIT wrapper that compiles element-wise array functions.

The ``Wrapper`` class turns a plain Python function whose body consists of
element-wise array assignments into a JIT-compiled kernel that runs in-place on
numpy arrays.  Compilation traces the function by calling it once with ``_Probe``
objects: every arithmetic operation builds a :class:`pylat.expr` expression tree
and every in-place update (``a += ...`` or ``a[:] = ...``) is recorded as an
:class:`pylat.expr.AssignExpr`.  The recorded assignments are then compiled with
:class:`pylat.jit.compile.JitCompiler` and cached per (dtype, rank) signature,
array layout and compile-time argument values.  Because the function is traced by
calling it, it never needs to be source-inspectable: closures, lambdas and
functions defined in interactive sessions all work.
"""

import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from inspect import Parameter
from typing import Any, Literal

import numpy as np
from llvmlite import binding as llvm
from typing_extensions import override

from ..expr import (
    AssignExpr,
    Cos,
    Exp,
    Expr,
    Int,
    Ln,
    Power,
    Rational,
    Roll,
    Sin,
    Slice,
    Symbol,
    Times,
    symbol,
)
from .argpass import ComplexFloatType, FloatType, IntType, LowerType, TypeContext
from .backend import Backend
from .compile import CompiledWrapper, JitCompiler, StandardLayoutMode
from .openmp import OpenMPBackend

_FUNC_MAP = {
    'sin': Sin,
    'cos': Cos,
    'exp': Exp,
    'log': Ln,
}

_BIN_UFUNCS = {
    'add': lambda l, r: l + r,
    'subtract': lambda l, r: l - r,
    'multiply': lambda l, r: l * r,
    'divide': lambda l, r: l / r,
    'true_divide': lambda l, r: l / r,
    'power': lambda l, r: l ** r,
}


def _as_expr(value) -> Expr:
    """Convert a probe or a Python/numpy constant into an ``Expr``."""
    if isinstance(value, _Probe):
        return value._expr
    if isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.floating):
        value = float(value)
    try:
        return Expr.as_expr(value)
    except ValueError as e:
        raise TypeError(f"unsupported operand of type {type(value).__name__} in jitted function") from e


class _Trace:
    """Records the assignments performed while the function is being traced."""

    def __init__(self, names: Mapping[str, Symbol]) -> None:
        self._names = names
        self.assigns: list[AssignExpr] = []

    def make_param_probe(self, name: str) -> '_Probe':
        return _Probe(self, self._names[name])

    def record(self, target: '_Probe', op: str, value) -> None:
        if not isinstance(target._expr, Symbol):
            raise TypeError("cannot update an intermediate expression; assign to a parameter instead")
        self.assigns.append(AssignExpr(target._expr, _as_expr(value), op))


class _Probe:
    """Records an element-wise operation by building an ``Expr`` tree."""

    __slots__ = ('_expr', '_trace')

    def __init__(self, trace: _Trace, expr: Expr) -> None:
        self._trace = trace
        self._expr = expr

    def _new(self, expr: Expr) -> '_Probe':
        return _Probe(self._trace, expr)

    # --- binary arithmetic ------------------------------------------------
    def __add__(self, other):
        return self._new(self._expr + _as_expr(other))

    def __radd__(self, other):
        return self._new(_as_expr(other) + self._expr)

    def __sub__(self, other):
        return self._new(self._expr - _as_expr(other))

    def __rsub__(self, other):
        return self._new(_as_expr(other) - self._expr)

    def __mul__(self, other):
        return self._new(self._expr * _as_expr(other))

    def __rmul__(self, other):
        return self._new(_as_expr(other) * self._expr)

    def __truediv__(self, other):
        return self._new(self._expr / _as_expr(other))

    def __rtruediv__(self, other):
        return self._new(_as_expr(other) / self._expr)

    def __pow__(self, other):
        return self._new(self._expr ** _as_expr(other))

    def __rpow__(self, other):
        return self._new(_as_expr(other) ** self._expr)

    # --- unary arithmetic -------------------------------------------------
    def __neg__(self):
        return self._new(Times((Int(-1), self._expr)))

    def __pos__(self):
        return self

    # --- in-place updates (recorded as assignments) ----------------------
    def __iadd__(self, other):
        self._trace.record(self, '+', other)
        return self

    def __isub__(self, other):
        self._trace.record(self, '-', other)
        return self

    def __imul__(self, other):
        self._trace.record(self, '*', other)
        return self

    def __itruediv__(self, other):
        self._trace.record(self, '/', other)
        return self

    def __setitem__(self, key, value):
        self._trace.record(self, '', value)

    # --- indexing and slicing -------------------------------------------
    def __getitem__(self, key):
        expr = self._expr
        if not isinstance(key, tuple):
            key = (key,)
        fixed: list[tuple[int, int]] = []
        for axis, k in enumerate(key):
            if isinstance(k, slice):
                if k.start is None and k.stop is None and k.step is None:
                    continue
                raise TypeError(
                    "range slices are not supported in jitted functions; "
                    "use an integer index or ':'"
                )
            if k is Ellipsis:
                raise TypeError("'...' is not supported in jitted functions; use explicit ':'")
            if isinstance(k, (int, np.integer)):
                fixed.append((axis, int(k)))
                continue
            raise TypeError(f"unsupported slice index {k!r}")
        # fix the highest axis first (innermost), so that lower axes stay valid
        # after the higher ones have been removed
        for axis, index in reversed(fixed):
            expr = Slice(expr, axis, index)
        return self._new(expr)

    # --- numpy functions (np.roll, ...) ---------------------------------
    def __array_function__(self, func, types, args, kwargs):
        if func is np.roll:
            return self._np_roll(*args, **kwargs)
        return NotImplemented

    def _np_roll(self, array, shift, axis=None):
        if not isinstance(array, _Probe):
            raise TypeError("np.roll requires a traced array in jitted functions")
        if isinstance(shift, _Probe) or isinstance(axis, _Probe):
            raise TypeError("np.roll shift and axis must be constants in jitted functions")
        if axis is None:
            raise TypeError("np.roll requires an explicit axis in jitted functions")
        axes = axis if isinstance(axis, (tuple, list)) else (axis,)
        shifts = shift if isinstance(shift, (tuple, list)) else (shift,) * len(axes)
        if len(axes) != len(shifts):
            raise TypeError("np.roll shift and axis must have the same length")
        expr = array._expr
        for ax, sh in zip(axes, shifts):
            expr = Roll(expr, int(ax), int(sh))
        return self._new(expr)

    # --- numpy ufuncs (np.sin, np.cos, np.exp, np.log, np.sqrt, ...) -----
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__' or len(kwargs) > 0:
            return NotImplemented
        name = ufunc.__name__
        if len(inputs) == 1:
            arg = inputs[0]
            if not isinstance(arg, _Probe):
                return NotImplemented
            if name == 'negative':
                return self._new(Times((Int(-1), arg._expr)))
            if name == 'positive':
                return self._new(arg._expr)
            if name == 'sqrt':
                return self._new(Power(arg._expr, Rational(1, 2)))
            fn = _FUNC_MAP.get(name)
            if fn is None:
                raise TypeError(f"unsupported numpy function {name!r} in jitted function")
            return self._new(fn(arg._expr))
        if len(inputs) == 2:
            # e.g. np.float64(3) * probe: numpy dispatches the operation here
            lhs, rhs = inputs
            if not (isinstance(lhs, _Probe) or isinstance(rhs, _Probe)):
                return NotImplemented
            l = lhs._expr if isinstance(lhs, _Probe) else _as_expr(lhs)
            r = rhs._expr if isinstance(rhs, _Probe) else _as_expr(rhs)
            fn = _BIN_UFUNCS.get(name)
            if fn is None:
                raise TypeError(f"unsupported numpy function {name!r} in jitted function")
            return self._new(fn(l, r))
        return NotImplemented

    # --- operations that make no sense on a traced value ------------------
    def __bool__(self) -> bool:
        raise TypeError("branching on a traced value is not supported in jitted functions")

    def __float__(self) -> float:
        raise TypeError("converting a traced value to a Python float is not supported in jitted functions")

    def __int__(self) -> int:
        raise TypeError("converting a traced value to a Python int is not supported in jitted functions")

    def __complex__(self) -> complex:
        raise TypeError("converting a traced value to a Python complex is not supported in jitted functions")

    def __index__(self) -> int:
        raise TypeError("using a traced value as an index is not supported in jitted functions")

    def __eq__(self, other) -> bool:
        raise TypeError("comparisons are not supported in jitted functions")

    def __ne__(self, other) -> bool:
        raise TypeError("comparisons are not supported in jitted functions")

    def __lt__(self, other) -> bool:
        raise TypeError("comparisons are not supported in jitted functions")

    def __le__(self, other) -> bool:
        raise TypeError("comparisons are not supported in jitted functions")

    def __gt__(self, other) -> bool:
        raise TypeError("comparisons are not supported in jitted functions")

    def __ge__(self, other) -> bool:
        raise TypeError("comparisons are not supported in jitted functions")


def _collect_symbols(expr: Expr) -> set[Symbol]:
    used: set[Symbol] = set()
    todo = [expr]
    while todo:
        elem = todo.pop()
        if isinstance(elem, Symbol):
            used.add(elem)
        else:
            todo.extend(elem.subexpressions())
    return used


def _determine_layout(values) -> StandardLayoutMode:
    """Pick the kernel layout that matches the actual array arguments.

    Returns ``ROW_MAJOR`` when every array is C-contiguous, ``COLUMN_MAJOR`` when
    every array is F-contiguous, and ``NONE`` (generic kernel) otherwise.
    """
    row_ok = True
    col_ok = True
    for value in values:
        if not isinstance(value, np.ndarray):
            continue
        if not value.flags['C_CONTIGUOUS']:
            row_ok = False
        if not value.flags['F_CONTIGUOUS']:
            col_ok = False
    if row_ok:
        return StandardLayoutMode.ROW_MAJOR
    if col_ok:
        return StandardLayoutMode.COLUMN_MAJOR
    return StandardLayoutMode.NONE


ParamKind = Literal['arg', 'varargs', 'kwargs']

def _infer_params(fn: Callable) -> tuple[tuple[str, ParamKind], ...]:
    try:
        signature = inspect.signature(fn)
    except ValueError as e:
        raise TypeError(f"cannot determine the signature of {fn!r}") from e
    params: list[tuple[str, ParamKind]] = []
    for name, p in signature.parameters.items():
        if p.kind is Parameter.KEYWORD_ONLY:
            raise TypeError(f"keyword-only parameter {name!r} is not supported")
        if p.default is not inspect.Parameter.empty:
            raise TypeError(f"default value on parameter {name!r} is not supported")
        if p.kind is Parameter.VAR_POSITIONAL:
            kind: ParamKind = 'varargs'
        elif p.kind is Parameter.VAR_KEYWORD:
            kind = 'kwargs'
        else:
            kind = 'arg'
        params.append((name, kind))
    return tuple(params)

class SignatureNode:
    pass

@dataclass(frozen=True)
class ArrayArgNode(SignatureNode):
    dtype: LowerType
    rank: int

    @override
    def __str__(self) -> str:
        return f"Array[{self.dtype}, {self.rank}]"

@dataclass(frozen=True)
class ComptimeValueArgNode(SignatureNode):
    value: Any

    @override
    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class TupleArgNode(SignatureNode):
    elements: tuple[SignatureNode, ...]

    @override
    def __str__(self) -> str:
        return f"Tuple[{', '.join(str(e) for e in self.elements)}]"

@dataclass(frozen=True)
class DictArgNode(SignatureNode):
    values: frozenset[tuple[str, SignatureNode]]

    @override
    def __str__(self) -> str:
        return f"Dict[{', '.join(f'{k} -> {v}' for k, v in self.values)}]"

@dataclass(frozen=True)
class ScalarArgNode(SignatureNode):
    dtype: LowerType

    @override
    def __str__(self) -> str:
        return f"Scalar[{self.dtype}]"

@dataclass(frozen=True)
class _JitCacheKey:
    signature: tuple[tuple[str, SignatureNode], ...]
    layout: StandardLayoutMode

class _JittedFunction:
    """The callable produced by ``Wrapper.jit``."""

    def __init__(self, wrapper: 'Wrapper', fn: Callable, params: tuple[tuple[str, ParamKind], ...], comptime_args: set[str | int] | None = None) -> None:
        self._wrapper = wrapper
        self._fn = fn
        self._params = params
        self._names = {name: symbol(name) for name, _ in params}
        self._param_positions = {
            name: index
            for index, (name, kind) in enumerate(params)
            if kind == 'arg'
        }
        self._comptime_args = comptime_args
        self._cache: dict[_JitCacheKey, tuple[CompiledWrapper, set[Symbol]]] = {}
        self.__name__ = getattr(fn, '__name__', 'jitted')
        self.__doc__ = getattr(fn, '__doc__', None)

    def _trace(self, key: _JitCacheKey) -> tuple[list[AssignExpr], set[Symbol]]:
        # compile-time arguments are traced with their constant value, so they are
        # baked into the expression trees (which also enables compile-time control
        # flow); the other arguments are traced with probes
        trace = _Trace(self._names)
        positional: list[Any] = []
        keyword: dict[str, Any] = {}
        for name, node in key.signature:
            if isinstance(node, ComptimeValueArgNode):
                positional.append(node.value)
            elif isinstance(node, TupleArgNode):
                for i, elem in enumerate(node.elements):
                    elem_name = f"{name}[{i}]"
                    if isinstance(elem, ComptimeValueArgNode):
                        positional.append(elem.value)
                    else:
                        positional.append(_Probe(trace, symbol(elem_name)))
            elif isinstance(node, DictArgNode):
                for kw_name, elem in node.values:
                    if isinstance(elem, ComptimeValueArgNode):
                        keyword[kw_name] = elem.value
                    else:
                        keyword[kw_name] = _Probe(trace, symbol(kw_name))
            else:
                positional.append(trace.make_param_probe(name))
        result = self._fn(*positional, **keyword)
        if result is not None:
            raise TypeError(
                f"{self.__name__}() must not return values; mutate the input arrays in place "
                "(e.g. with += or a[:] = ...)"
            )
        if len(trace.assigns) == 0:
            raise TypeError(f"{self.__name__}() must contain at least one in-place assignment")
        used: set[Symbol] = set()
        for assign in trace.assigns:
            used |= _collect_symbols(assign.lhs)
            used |= _collect_symbols(assign.rhs)
        return trace.assigns, used

    def _infer_arg_type(self, value) -> tuple[LowerType, int]:
        if isinstance(value, _Probe):
            raise TypeError(
                f"{self.__name__}() cannot be called from within another jitted function; "
                "use a plain helper function instead"
            )
        if isinstance(value, np.ndarray):
            return LowerType.from_numpy_dtype(str(value.dtype)), value.ndim
        if isinstance(value, (np.floating, float)):
            return self._wrapper._real_type, 0
        if isinstance(value, (np.complexfloating, complex)):
            return ComplexFloatType(self._wrapper._real_type), 0
        if isinstance(value, (np.integer, int)):
            return self._wrapper._index_type, 0
        raise TypeError(f"unsupported argument type: {type(value).__name__}")

    def _is_comptime_arg(self, name: str, value) -> bool:
        """Whether an argument is a compile-time constant: either explicitly declared
        in ``comptime_args`` (by name or parameter position), or an unsupported runtime
        type that is hashable (e.g. tuples used with ``np.roll``)."""
        index = self._param_positions.get(name)
        if index is not None and self._comptime_args is not None and (name in self._comptime_args or index in self._comptime_args):
            try:
                hash(value)
            except TypeError as e:
                raise TypeError(
                    f"comptime argument {name!r} must be hashable (it is part of the JIT cache key)"
                ) from e
            return True
        if isinstance(value, _Probe):
            # nested jitted calls are rejected by _infer_arg_type with a clear message
            return False
        try:
            self._infer_arg_type(value)
        except TypeError:
            try:
                hash(value)
            except TypeError as e:
                raise TypeError(
                    f"unsupported argument type: {type(value).__name__}; pass an array, a "
                    "scalar, or a hashable compile-time constant"
                ) from e
            return True
        return False

    def _classify(self, name: str, value) -> SignatureNode:
        """Classify a bound argument into its signature node for the cache key."""
        if self._is_comptime_arg(name, value):
            return ComptimeValueArgNode(value)
        lower_type, dim = self._infer_arg_type(value)
        if dim == 0:
            return ScalarArgNode(lower_type)
        return ArrayArgNode(lower_type, dim)

    def _bind_args(self, args, kwargs) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], dict[str, Any]]:
        """Bind the call arguments to the formal parameters.

        Returns ``(fixed, variadic, keyword)``: ``fixed`` and ``variadic`` are
        ``(name, value)`` pairs for the fixed positional parameters (named with the
        parameter name) and the ``*varargs`` elements (named ``<param>[i]``);
        ``keyword`` maps keyword names to values for the ``**kwargs`` parameter.
        """
        fixed = [name for name, kind in self._params if kind == 'arg']
        varargs_name = next((name for name, kind in self._params if kind == 'varargs'), None)
        kwargs_name = next((name for name, kind in self._params if kind == 'kwargs'), None)
        if len(args) < len(fixed):
            raise TypeError(
                f"{self.__name__}() missing {len(fixed) - len(args)} required positional argument(s)"
            )
        fixed_args = list(zip(fixed, args))
        rest = args[len(fixed):]
        if varargs_name is None:
            if len(rest) > 0:
                raise TypeError(
                    f"{self.__name__}() takes {len(fixed)} positional arguments but {len(args)} were given"
                )
            variadic: list[tuple[str, Any]] = []
        else:
            variadic = [(f"{varargs_name}[{i}]", value) for i, value in enumerate(rest)]
        if kwargs_name is None:
            if len(kwargs) > 0:
                raise TypeError(
                    f"{self.__name__}() got an unexpected keyword argument {next(iter(kwargs))!r}"
                )
            keyword: dict[str, Any] = {}
        else:
            keyword = dict(kwargs)
        return fixed_args, variadic, keyword

    def _build_signature(self, fixed: list[tuple[str, Any]], variadic: list[tuple[str, Any]], keyword: dict[str, Any]) -> tuple[tuple[str, SignatureNode], ...]:
        """Build the cache-key signature: one ``(name, node)`` entry per formal
        parameter, in declaration order.  ``*varargs`` elements are collected into a
        :class:`TupleArgNode` and ``**kwargs`` values into a :class:`DictArgNode`,
        so the signature determines the runtime argument types, the compile-time
        values and how the call arguments map to symbols."""
        signature: list[tuple[str, SignatureNode]] = [
            (name, self._classify(name, value)) for name, value in fixed
        ]
        varargs_name = next((name for name, kind in self._params if kind == 'varargs'), None)
        kwargs_name = next((name for name, kind in self._params if kind == 'kwargs'), None)
        if varargs_name is not None:
            signature.append((varargs_name, TupleArgNode(tuple(self._classify(n, v) for n, v in variadic))))
        if kwargs_name is not None:
            signature.append((kwargs_name, DictArgNode(frozenset((n, self._classify(n, v)) for n, v in keyword.items()))))
        return tuple(signature)

    def _iter_runtime_args(self, key: _JitCacheKey) -> Iterator[tuple[str, LowerType, int]]:
        """Yield ``(name, lower_type, dim)`` for every runtime argument in the key."""
        def dim_of(node: SignatureNode) -> int:
            return node.rank if isinstance(node, ArrayArgNode) else 0
        for name, node in key.signature:
            if isinstance(node, (ArrayArgNode, ScalarArgNode)):
                yield name, node.dtype, dim_of(node)
            elif isinstance(node, TupleArgNode):
                for i, elem in enumerate(node.elements):
                    if isinstance(elem, (ArrayArgNode, ScalarArgNode)):
                        yield f"{name}[{i}]", elem.dtype, dim_of(elem)
            elif isinstance(node, DictArgNode):
                for kw_name, elem in node.values:
                    if isinstance(elem, (ArrayArgNode, ScalarArgNode)):
                        yield kw_name, elem.dtype, dim_of(elem)

    def _compile(self, key: _JitCacheKey, assigns: list[AssignExpr]) -> CompiledWrapper:
        context = TypeContext()
        for name, lower_type, dim in self._iter_runtime_args(key):
            context.set_symbol(self._names.get(name, symbol(name)), lower_type, dim)
        compiler = JitCompiler(
            self._wrapper._backend,
            real_type=self._wrapper._real_type,
            index_type=self._wrapper._index_type,
        )
        return compiler.compile_assignments(assigns, context, standard_layout=key.layout)

    def __call__(self, *args, **kwargs):
        fixed, variadic, keyword = self._bind_args(args, kwargs)
        signature = self._build_signature(fixed, variadic, keyword)
        layout = _determine_layout(
            [v for _, v in fixed] + [v for _, v in variadic] + list(keyword.values())
        )
        key = _JitCacheKey(signature=signature, layout=layout)
        entry = self._cache.get(key)
        if entry is None:
            assigns, used = self._trace(key)
            compiled = self._compile(key, assigns)
            entry = (compiled, used)
            self._cache[key] = entry
        compiled, used = entry
        arg_map: dict[Symbol, Any] = {}
        for name, value in fixed + variadic:
            sym = symbol(name)
            if sym in used:
                arg_map[sym] = value
        for name, value in keyword.items():
            sym = symbol(name)
            if sym in used:
                arg_map[sym] = value
        return compiled.call(arg_map)

    def print_all(self):
        """Print the LLVM IR of the most recently compiled kernel."""
        if len(self._cache) == 0:
            return []
        compiled, _ = list(self._cache.values())[-1]
        return compiled.print_all()

    def __repr__(self) -> str:
        return f"<jitted {self.__name__}>"


class Wrapper:
    """
    Usage example:

    ```python
    wrapper = Wrapper()

    @wrapper.jit()
    def my_func(a, b, c, dt):
        a += c * dt
        b += c * dt + c * 2

    a = np.random.rand(8, 9, 10)
    b = np.random.rand(8, 9, 10)
    c = np.random.rand(8, 9, 10)
    a0 = a.copy()
    b0 = b.copy()
    dt = 0.5
    my_func(a, b, c, dt)
    assert np.allclose(a, a0 + c * dt)
    assert np.allclose(b, b0 + c * dt + c * 2)
    ```
    """

    def __init__(self, backend: Backend | None = None, real_type: FloatType | None = None, index_type: IntType | None = None) -> None:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self._backend = backend if backend is not None else OpenMPBackend()
        self._real_type = real_type if real_type is not None else FloatType(64)
        self._index_type = index_type if index_type is not None else IntType(64, False)

    def jit(self, fn: Callable | None = None, comptime_args: set[str | int] | None = None):
        """Decorator that compiles a function of element-wise array assignments.

        ``comptime_args`` lists parameters whose values are baked into the kernel as
        compile-time constants instead of being passed as runtime scalars: match them
        by name (``str``) or by position (``int``).  Arguments that cannot be passed
        as runtime values and are hashable are also treated as compile-time.
        """
        def decorator(f: Callable) -> _JittedFunction:
            return _JittedFunction(self, f, _infer_params(f), comptime_args)
        if fn is not None:
            return decorator(fn)
        return decorator
