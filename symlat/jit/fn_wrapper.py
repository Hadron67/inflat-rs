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

import ctypes
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter
from typing import Any, Literal, overload

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
    Sum,
    Symbol,
    Times,
)
from .backend import Backend
from .compile import CompiledWrapper, JitCompiler, StandardLayoutMode
from .openmp import OpenMPBackend
from .type import (
    ComplexFloatType,
    FloatType,
    IntType,
    LowerType,
    SymbolShape,
    SymbolTypeDesc,
    TypeResolver,
)

# placeholder symbol namespace for reduction results; ``Sum`` nodes in the traced
# assignments are replaced by symbols in this namespace before compilation
_SUM_PREFIX = ('__sum',)

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

    def __init__(self) -> None:
        self.assigns: list[AssignExpr] = []
        # each ``np.sum`` in the traced body is recorded here as a placeholder
        # symbol (in first-seen order); the reductions are compiled from this
        # registry and the placeholders take their place in the expressions
        self.sums: dict[Sum, Symbol] = {}

    def sum_placeholder(self, sum_node: Sum) -> Symbol:
        """Return the placeholder symbol assigned to a ``Sum`` node, creating
        one (and registering the node) on first use."""
        sum_node = Sum(sum_node.expr.normalize())
        if sum_node not in self.sums:
            self.sums[sum_node] = Symbol(_SUM_PREFIX + (str(len(self.sums)),))
        return self.sums[sum_node]

    def make_param_probe(self, name: tuple[str, ...], rank: int) -> '_Probe':
        return _Probe(self, Symbol(name), rank)

    def record(self, target: '_Probe', op: str, value) -> None:
        if not isinstance(target._expr, Symbol):
            raise TypeError("cannot update an intermediate expression; assign to a parameter instead")
        self.assigns.append(AssignExpr(target._expr, _as_expr(value), op))


class _Probe:
    """Records an element-wise operation by building an ``Expr`` tree."""

    __slots__ = ('_expr', '_ndim', '_trace')

    def __init__(self, trace: _Trace, expr: Expr, ndim: int | None = None) -> None:
        self._trace = trace
        self._expr = expr
        self._ndim = ndim

    def _new(self, expr: Expr) -> '_Probe':
        return _Probe(self._trace, expr)

    @property
    def ndim(self) -> int:
        if self._ndim is None:
            raise ValueError("ndim is not set")
        return self._ndim

    @property
    def shape(self) -> tuple[Expr, ...]:
        if self._ndim is not None and isinstance(self._expr, Symbol):
            return tuple(SymbolShape(self._expr, i) for i in range(self._ndim))
        raise ValueError("shape is not available for this expression")

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

    # --- numpy functions (np.roll, np.sum, ...) --------------------------
    def __array_function__(self, func, types, args, kwargs):
        if func is np.roll:
            return self._np_roll(*args, **kwargs)
        if func is np.sum:
            return self._np_sum(*args, **kwargs)
        return NotImplemented

    def _np_sum(self, array, **kwargs):
        if not isinstance(array, _Probe):
            raise TypeError("np.sum requires a traced array in jitted functions")
        if kwargs.get('axis', None) is not None:
            raise TypeError("only np.sum over all axes is supported in jitted functions")
        unsupported = [name for name, value in kwargs.items()
                       if name != 'axis' and value is not None and not isinstance(value, bool)]
        if unsupported:
            raise TypeError(f"unsupported np.sum argument(s): {unsupported} in jitted functions")
        if isinstance(array._expr, Sum) or array._expr in self._trace.sums.values():
            raise TypeError("nested np.sum is not supported in jitted functions")
        # the compiled kernels cannot lower a Sum node, so it is replaced by a
        # placeholder scalar symbol now and compiled as a reduction later
        return self._new(self._trace.sum_placeholder(Sum(array._expr)))

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
    runtime_arg_pos: int

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
    type: type = dict

    @override
    def __str__(self) -> str:
        return f"Dict[{', '.join(f'{k} -> {v}' for k, v in self.values)}, type={self.type}]"

@dataclass(frozen=True)
class ScalarArgNode(SignatureNode):
    dtype: LowerType
    runtime_arg_pos: int
    is_ref: bool = False

    @override
    def __str__(self) -> str:
        if self.is_ref:
            return f"ScalarRef[{self.dtype}]"
        return f"Scalar[{self.dtype}]"

@dataclass(frozen=True)
class Signature:
    fixed_args: tuple[SignatureNode, ...]
    varargs: TupleArgNode | None
    kwargs: DictArgNode | None

    def all_nodes(self) -> tuple[SignatureNode, ...]:
        ret = self.fixed_args
        if self.varargs is not None:
            ret += (self.varargs,)
        if self.kwargs is not None:
            ret += (self.kwargs,)
        return ret

@dataclass(frozen=True)
class _JitCacheKey:
    signature: Signature
    layout: StandardLayoutMode

def _gen_fill_one_arg(snode: SignatureNode, value_str: str, runtime_args: dict[int, str]):
    """Record the runtime arguments reachable through one signature node.

    ``value_str`` is a Python expression (e.g. ``args[3][0]``) that yields the
    argument's value at kernel-call time; the compiled kernel receives the
    runtime arguments in the order of their ``runtime_arg_pos``."""
    todo = [(snode, value_str)]
    while todo:
        snode, value_str = todo.pop()
        match snode:
            case ScalarArgNode() | ArrayArgNode():
                assert snode.runtime_arg_pos not in runtime_args
                runtime_args[snode.runtime_arg_pos] = value_str
            case ComptimeValueArgNode():
                pass  # baked into the expression trees, not passed at runtime
            case TupleArgNode():
                for child, item in zip(snode.elements, (f"{value_str}[{i}]" for i in range(len(snode.elements)))):
                    todo.append((child, item))
            case DictArgNode():
                for k, v in snode.values:
                    if snode.type is dict:
                        todo.append((v, f'{value_str}["{k}"]'))
                    else:
                        todo.append((v, f'getattr({value_str}, "{k}")'))
            case _:
                raise ValueError(f"Unexpected node type: {snode}")

def _gen_args_converter(signature: Signature) -> Callable:
    """Generate a ``__invoke(wrappers, args)`` function that unpacks the raw
    call arguments and drives the compiled kernels.

    ``args`` holds one element per signature entry (``Signature.all_nodes``):
    the fixed values, then the variadic list and the keyword dict (when such
    parameters exist).  ``wrappers`` is ``(main, *sums)``: the sum kernels are
    called first with the runtime arguments, and their scalar results are passed
    to the main kernel after the runtime arguments.  The generated function is
    specialised per signature, so the (positional) unpacking done per call
    reduces to plain indexing."""
    runtime_args: dict[int, str] = {}
    for i, snode in enumerate(signature.all_nodes()):
        _gen_fill_one_arg(snode, f'args[{i}]', runtime_args)
    exprs = [runtime_args[i] for i in range(len(runtime_args))]

    fname = '__invoke'
    lines = [
        f"def {fname}(wrappers, args):",
        "    main = wrappers[0]",
        "    sums = wrappers[1:]",
        *[f"    v{i} = {expr}" for i, expr in enumerate(exprs)],
        f"    svals = [s.call({', '.join(f'v{i}' for i in range(len(exprs)))}) for s in sums]",
        f"    return main.call({', '.join([*(f'v{i}' for i in range(len(exprs))), '*svals'])})",
    ]
    globals = {}
    exec('\n'.join(lines), globals)  # noqa: S102
    return globals[fname]

def _create_one_probe_arg(trace: _Trace, snode: SignatureNode, name: tuple[str, ...]) -> Any:
    match snode:
        case ArrayArgNode():
            return trace.make_param_probe(name, snode.rank)
        case ScalarArgNode():
            return trace.make_param_probe(name, 0)
        case ComptimeValueArgNode():
            return snode.value
        case TupleArgNode():
            return tuple(_create_one_probe_arg(trace, elem, name + (str(i),)) for i, elem in enumerate(snode.elements))
        case DictArgNode():
            values = {k: _create_one_probe_arg(trace, v, name + (k,)) for k, v in snode.values}
            if snode.type is dict:
                return values
            # rebuild the object with probe attributes so that attribute access in
            # the traced body records operations on the individual fields
            obj = object.__new__(snode.type)
            obj.__dict__.update(values)
            return obj
        case _:
            raise TypeError(f"unexpected signature node type: {snode}")

def _create_probe_args(trace: _Trace, signature: Signature, args_info: '_FormalArgsInfo') -> tuple[list[Any], dict[str, Any]]:
    """Build the arguments used to trace the function body.

    Compile-time values are passed as-is (baking them into the expression trees,
    which also enables compile-time control flow); the other arguments are passed
    as probes that record the operations performed on them."""
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    ns = '__trace'
    for name, node in zip(args_info.fixed_names, signature.fixed_args):
        positional.append(_create_one_probe_arg(trace, node, (ns, name)))
    if signature.varargs is not None:
        name = args_info.varargs_name
        assert name is not None
        # *varargs elements are passed as separate positional arguments
        positional.extend(_create_one_probe_arg(trace, signature.varargs, (ns, name)))
    if signature.kwargs is not None:
        name = args_info.kwargs_name
        assert name is not None
        # **kwargs values are passed as keyword arguments
        keyword.update(_create_one_probe_arg(trace, signature.kwargs, (ns, name)))
    return positional, keyword


class _FormalArgsInfo:
    """Description of a jitted function's formal parameters.

    Holds the parameter list inferred from the source function together with the
    declared compile-time arguments, and answers how call arguments bind to the
    parameters.
    """

    def __init__(self, params: tuple[tuple[str, ParamKind], ...], comptime_args: set[str | int] | None = None) -> None:
        self._params = params
        self._param_positions = {
            name: index
            for index, (name, kind) in enumerate(params)
            if kind == 'arg'
        }
        self._comptime_poses: set[int] = set()
        if comptime_args is not None:
            for arg in comptime_args:
                if isinstance(arg, int):
                    self._comptime_poses.add(arg)
                elif isinstance(arg, str) and arg in self._param_positions:
                    self._comptime_poses.add(self._param_positions[arg])

    @property
    def fixed_names(self) -> list[str]:
        """Names of the fixed positional parameters, in declaration order."""
        return [name for name, kind in self._params if kind == 'arg']

    @property
    def varargs_name(self) -> str | None:
        """Name of the ``*varargs`` parameter, or ``None`` when there is none."""
        return next((name for name, kind in self._params if kind == 'varargs'), None)

    def has_varargs(self) -> bool:
        """Whether the function has a ``*varargs`` parameter."""
        return self.varargs_name is not None

    def has_kwargs(self) -> bool:
        """Whether the function has a ``**kwargs`` parameter."""
        return self.kwargs_name is not None

    @property
    def kwargs_name(self) -> str | None:
        """Name of the ``**kwargs`` parameter, or ``None`` when there is none."""
        return next((name for name, kind in self._params if kind == 'kwargs'), None)

    def is_explicit_comptime(self, pos: int) -> bool:
        """Whether a fixed positional parameter is declared in ``comptime_args``,
        matched by name or by parameter position."""
        return pos in self._comptime_poses

    def bind_args(self, args, kwargs, func_name: str) -> tuple[list[Any], list[Any], dict[str, Any]]:
        """Bind the call arguments to the formal parameters.

        Returns fixed_args, var_args, kwargs.
        """
        fixed = self.fixed_names
        varargs_name = self.varargs_name
        kwargs_name = self.kwargs_name
        if len(args) < len(fixed):
            raise TypeError(
                f"{func_name}() missing {len(fixed) - len(args)} required positional argument(s)"
            )
        fixed_args = list(args[:len(fixed)])
        rest = args[len(fixed):]
        if varargs_name is None:
            if len(rest) > 0:
                raise TypeError(
                    f"{func_name}() takes {len(fixed)} positional arguments but {len(args)} were given"
                )
            variadic: list[Any] = []
        else:
            variadic = list(rest)
        if kwargs_name is None:
            if len(kwargs) > 0:
                raise TypeError(
                    f"{func_name}() got an unexpected keyword argument {next(iter(kwargs))!r}"
                )
            keyword: dict[str, Any] = {}
        else:
            keyword = dict(kwargs)
        return fixed_args, variadic, keyword

class _JittedFunction:
    """The callable produced by ``Wrapper.jit``."""

    def __init__(self, wrapper: 'Wrapper', fn: Callable, params: tuple[tuple[str, ParamKind], ...], comptime_args: set[str | int] | None = None) -> None:
        self._wrapper = wrapper
        self._fn = fn
        self._args_info = _FormalArgsInfo(params, comptime_args)
        self._cache: dict[_JitCacheKey, tuple[tuple[CompiledWrapper, ...], Callable]] = {}
        self.__name__ = getattr(fn, '__name__', 'jitted')
        self.__doc__ = getattr(fn, '__doc__', None)

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> '_JittedFunction': ...

    @overload
    def __get__(self, instance: Any, owner: type | None = None) -> Callable[..., Any]: ...

    def __get__(self, instance: Any, owner: type | None = None) -> '_JittedFunction | Callable[..., Any]':
        """Support jitted methods: ``obj.f(...)`` binds ``obj`` as the first
        argument (which is then inlined as a :class:`DictArgNode`)."""
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)

    def _trace(self, key: _JitCacheKey) -> tuple[list[AssignExpr], dict[Sum, Symbol]]:
        """Trace the function body; returns the recorded assignments together
        with the ``Sum`` -> placeholder symbol registry (in first-seen order).

        Compile-time arguments are traced with their constant value, so they are
        baked into the expression trees (which also enables compile-time control
        flow); the other arguments are traced with probes."""
        trace = _Trace()
        positional, keyword = _create_probe_args(trace, key.signature, self._args_info)
        result = self._fn(*positional, **keyword)
        if result is not None:
            raise TypeError(
                f"{self.__name__}() must not return values; mutate the input arrays in place "
                "(e.g. with += or a[:] = ...)"
            )
        if len(trace.assigns) == 0:
            raise TypeError(f"{self.__name__}() must contain at least one in-place assignment")
        return trace.assigns, trace.sums

    def _infer_arg_type(self, value) -> tuple[LowerType, int, bool]:
        """Infer ``(lower_type, dimension, is_ref)`` for a runtime argument.

        ``is_ref`` marks address-takable scalars (0-d arrays and ctypes scalars
        like ``c_double``/``c_int``): they can be passed by reference so that
        the kernel can write back to them.  numpy scalars (``np.float64`` etc.)
        and Python scalars have no writable address and are always by value."""
        if isinstance(value, _Probe):
            raise TypeError(
                f"{self.__name__}() cannot be called from within another jitted function; "
                "use a plain helper function instead"
            )
        if isinstance(value, np.ndarray):
            return LowerType.from_numpy_dtype(str(value.dtype)), value.ndim, value.ndim == 0
        if isinstance(value, (np.floating, float)):
            return self._wrapper._real_type, 0, False
        if isinstance(value, (np.complexfloating, complex)):
            return ComplexFloatType(self._wrapper._real_type), 0, False
        if isinstance(value, (np.integer, int)):
            return self._wrapper._index_type, 0, False
        if isinstance(value, ctypes._SimpleCData):
            return LowerType.from_numpy_dtype(str(np.dtype(type(value)))), 0, True
        raise TypeError(f"unsupported argument type: {type(value).__name__}")

    def _is_comptime_arg(self, pos: int, value) -> bool:
        """Whether an argument is a compile-time constant: either explicitly declared
        in ``comptime_args`` (by name or parameter position), or an unsupported runtime
        type that is hashable (e.g. tuples used with ``np.roll``).  Objects with
        instance attributes are supported (they are inlined) and are never
        compile-time."""
        if self._args_info.is_explicit_comptime(pos):
            try:
                hash(value)
            except TypeError as e:
                raise TypeError(
                    f"comptime argument at position {pos} must be hashable (it is part of the JIT cache key)"
                ) from e
            return True
        if isinstance(value, _Probe):
            # nested jitted calls are rejected by _infer_arg_type with a clear message
            return False
        try:
            self._infer_arg_type(value)
        except TypeError:
            if getattr(value, '__dict__', None) is not None:
                # general objects with instance attributes are inlined instead of
                # being baked as compile-time constants
                return False
            try:
                hash(value)
            except TypeError as e:
                raise TypeError(
                    f"unsupported argument type: {type(value).__name__}; pass an array, a "
                    "scalar, or a hashable compile-time constant"
                ) from e
            return True
        return False

    def _classify(self, pos: int, value, runtime_arg_pos: int) -> tuple[SignatureNode, int]:
        """Classify a bound argument into its signature node for the cache key.

        ``pos`` is the formal parameter position (``-1`` for ``*varargs``
        elements, ``**kwargs`` values and inlined object attributes, which cannot
        be declared compile-time); ``runtime_arg_pos`` is the position of the
        argument in the compiled kernel's positional argument list; it is
        recorded on runtime nodes only.  Returns the node together with the
        number of runtime positions it consumes."""
        if self._is_comptime_arg(pos, value):
            return ComptimeValueArgNode(value), 0
        if isinstance(value, (np.ndarray, np.floating, np.complexfloating, np.integer, float, complex, int, ctypes._SimpleCData, _Probe)):
            # probes reach _infer_arg_type here and are rejected with a clear
            # "nested jitted call" error
            lower_type, dim, is_ref = self._infer_arg_type(value)
            if dim == 0:
                return ScalarArgNode(lower_type, runtime_arg_pos, is_ref), 1
            return ArrayArgNode(lower_type, dim, runtime_arg_pos), 1
        # general object: inline it by classifying each attribute recursively
        fields = getattr(value, '__dict__', None)
        if fields is not None:
            next_pos = runtime_arg_pos
            entries: set[tuple[str, SignatureNode]] = set()
            for name in sorted(fields):
                node, count = self._classify(-1, fields[name], next_pos)
                entries.add((name, node))
                next_pos += count
            return DictArgNode(frozenset(entries), type=type(value)), next_pos - runtime_arg_pos
        raise TypeError(f"unsupported argument type: {type(value).__name__}")

    def _build_signature(self, fixed: list[Any], variadic: list[Any], keyword: dict[str, Any]) -> Signature:
        """Build the cache-key signature from the bound call arguments.

        Fixed parameters become one node each, ``*varargs`` elements are
        collected into a :class:`TupleArgNode` and ``**kwargs`` values into a
        :class:`DictArgNode`, grouped in a :class:`Signature`.  The nodes
        determine the runtime argument types, the compile-time values and how the
        call arguments map to symbols; every runtime node records its position in
        the compiled kernel's positional argument list."""
        runtime_pos = 0

        def add(pos: int, value: Any) -> SignatureNode:
            nonlocal runtime_pos
            node, count = self._classify(pos, value, runtime_pos)
            runtime_pos += count
            return node

        fixed_nodes = tuple(add(i, value) for i, value in enumerate(fixed))
        varargs_node: SignatureNode | None = None
        if self._args_info.has_varargs():
            varargs_node = TupleArgNode(tuple(add(-1, v) for v in variadic))
        kwargs_node: SignatureNode | None = None
        if self._args_info.has_kwargs():
            # sort the keywords so the runtime positions are deterministic even
            # though DictArgNode stores them in an unordered frozenset
            kwargs_node = DictArgNode(frozenset((n, add(-1, v)) for n, v in sorted(keyword.items())))
        return Signature(fixed_args=fixed_nodes, varargs=varargs_node, kwargs=kwargs_node)

    def _runtime_args(self, key: _JitCacheKey) -> list[tuple[int, Symbol, LowerType, int, bool]]:
        """Return ``(runtime_arg_pos, symbol, lower_type, dim, is_ref)`` for every
        runtime argument in the key, using the positions recorded in the signature
        nodes.
        The traversal mirrors ``_create_one_probe_arg``: it descends into
        ``TupleArgNode``/``DictArgNode`` wherever they appear and builds symbols
        from the same probe paths used while tracing (the ``__trace`` namespace
        followed by the parameter path), so that they match the symbols recorded
        in the traced assignments."""
        def dim_of(node: SignatureNode) -> int:
            return node.rank if isinstance(node, ArrayArgNode) else 0

        def probe_symbol(path: tuple[str, ...]) -> Symbol:
            return Symbol(('__trace',) + path)

        result: list[tuple[int, Symbol, LowerType, int, bool]] = []
        todo: list[tuple[SignatureNode, tuple[str, ...]]] = []

        def walk(node: SignatureNode, path: tuple[str, ...]) -> None:
            todo.append((node, path))
            while todo:
                snode, spath = todo.pop()
                match snode:
                    case ArrayArgNode() | ScalarArgNode():
                        result.append((snode.runtime_arg_pos, probe_symbol(spath), snode.dtype, dim_of(snode), getattr(snode, 'is_ref', False)))
                    case ComptimeValueArgNode():
                        pass  # baked into the expression trees, not passed at runtime
                    case TupleArgNode():
                        todo.extend((elem, spath + (str(i),)) for i, elem in enumerate(snode.elements))
                    case DictArgNode():
                        todo.extend((v, spath + (k,)) for k, v in snode.values)
                    case _:
                        raise TypeError(f"unexpected signature node type: {snode}")

        signature = key.signature
        for name, node in zip(self._args_info.fixed_names, signature.fixed_args):
            walk(node, (name,))
        if signature.varargs is not None:
            varargs_name = self._args_info.varargs_name
            assert varargs_name is not None
            walk(signature.varargs, (varargs_name,))
        if signature.kwargs is not None:
            kwargs_name = self._args_info.kwargs_name
            assert kwargs_name is not None
            walk(signature.kwargs, (kwargs_name,))
        return result

    def _compile(self, key: _JitCacheKey) -> tuple[CompiledWrapper, tuple[CompiledWrapper, ...]]:
        """Compile the traced assignments into kernels.

        ``_FunctionCompiler`` cannot lower :class:`Sum` nodes, so every ``Sum``
        recorded while tracing is compiled as a separate reduction kernel and the
        traced assignments (which already reference the placeholder symbols)
        become the main kernel; the main kernel takes the reduction results as
        additional scalar arguments.  Returns ``(main, sums)`` where ``sums``
        holds one reduction kernel per ``Sum``, in the same order as the
        placeholders."""
        runtime = self._runtime_args(key)
        # sort by runtime_arg_pos: the walk order is not necessarily the position
        # order (the traversal uses a stack), and the converter passes values in
        # position order
        args_by_pos: dict[int, tuple[Symbol, SymbolTypeDesc]] = {
            pos: (sym, SymbolTypeDesc(lower_type, dim, is_ref)) for pos, sym, lower_type, dim, is_ref in runtime
        }
        args = [args_by_pos[i] for i in range(len(args_by_pos))]
        compiler = JitCompiler(
            self._wrapper._backend,
            real_type=self._wrapper._real_type,
            index_type=self._wrapper._index_type,
        )
        assigns, sums = self._trace(key)
        assigns = [a.normalize() for a in assigns]
        context = {sym: SymbolTypeDesc(lower_type, dim, is_ref) for _, sym, lower_type, dim, is_ref in runtime}
        # reduction placeholders are rank-0 scalars; include them so that the
        # shape resolver can handle assignments that read a reduction result
        for sym in sums.values():
            context[sym] = SymbolTypeDesc(self._wrapper._index_type, 0)
        resolver = TypeResolver(context, compiler)
        if len(sums) == 0:
            main = compiler.compile_assignments(args, assigns, standard_layout=key.layout)
            return main, ()
        placeholder_symbols = list(sums.values())
        sum_types: dict[Sum, LowerType] = {}
        for sum_node in sums:
            sum_type = resolver.get_type(sum_node.expr)
            if isinstance(sum_type, IntType):
                # integer sums follow the C convention of being signed
                sum_type = IntType(sum_type.bits, True)
            sum_types[sum_node] = sum_type
        sum_wrappers = tuple(
            compiler.compile_reduction(args, sum_node.expr, standard_layout=key.layout)
            for sum_node in sums
        )
        main = compiler.compile_assignments(
            args + [(sym, SymbolTypeDesc(sum_types[sum_node], 0)) for sum_node, sym in zip(sums, placeholder_symbols)],
            assigns,
            standard_layout=key.layout,
        )
        return main, sum_wrappers

    def __call__(self, *args, **kwargs):
        fixed, variadic, keyword = self._args_info.bind_args(args, kwargs, self.__name__)
        signature = self._build_signature(fixed, variadic, keyword)
        layout = _determine_layout(fixed + variadic + list(keyword.values()))
        key = _JitCacheKey(signature=signature, layout=layout)
        cached = self._cache.get(key)
        wrappers = None
        converter = None
        if cached is None:
            main, sums = self._compile(key)
            wrappers = (main, *sums)
            converter = _gen_args_converter(signature)
            self._cache[key] = wrappers, converter
        else:
            wrappers, converter = cached
        # the converter expects one element per signature entry: the fixed
        # values, then the variadic list and the keyword dict (when present)
        invoke_args: list[Any] = list(fixed)
        if self._args_info.has_varargs():
            invoke_args.append(variadic)
        if self._args_info.has_kwargs():
            invoke_args.append(keyword)
        return converter(wrappers, invoke_args)

    def print_all(self):
        """Print the LLVM IR of the most recently compiled kernel."""
        if len(self._cache) == 0:
            return []
        return list(self._cache.values())[-1][0][0].print_all()

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

    @overload
    def jit(self, fn: Callable, comptime_args: set[str | int] | None = None) -> _JittedFunction: ...
    @overload
    def jit(self, fn: None = None, comptime_args: set[str | int] | None = None) -> Callable[[Callable], _JittedFunction]: ...
    def jit(self, fn: Callable | None = None, comptime_args: set[str | int] | None = None) -> _JittedFunction | Callable[[Callable], _JittedFunction]:
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
