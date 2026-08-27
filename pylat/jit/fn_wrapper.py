"""User-facing JIT wrapper that compiles element-wise array functions.

The ``Wrapper`` class turns a plain Python function whose body consists of
element-wise array assignments into a JIT-compiled kernel that runs in-place on
numpy arrays.  Compilation traces the function by calling it once with ``_Probe``
objects: every arithmetic operation builds a :class:`pylat.expr` expression tree
and every in-place update (``a += ...`` or ``a[:] = ...``) is recorded as an
:class:`pylat.expr.AssignExpr`.  The recorded assignments are then compiled with
:class:`pylat.jit.compile.JitCompiler` and cached per (dtype, rank) signature of
the arguments.  Because the function is traced by calling it, it never needs to
be source-inspectable: closures, lambdas and functions defined in interactive
sessions all work.
"""

import inspect
from collections.abc import Callable, Mapping

import numpy as np
from llvmlite import binding as llvm

from ..expr import (
    AssignExpr,
    Cos,
    Exp,
    Expr,
    Int,
    Ln,
    Power,
    Rational,
    Sin,
    Symbol,
    Times,
    symbol,
)
from .argpass import ComplexFloatType, FloatType, IntType, LowerType, TypeContext
from .backend import Backend
from .compile import CompiledWrapper, JitCompiler
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


def _infer_params(fn: Callable) -> tuple[str, ...]:
    try:
        signature = inspect.signature(fn)
    except ValueError as e:
        raise TypeError(f"cannot determine the signature of {fn!r}") from e
    params: list[str] = []
    for name, p in signature.parameters.items():
        if p.kind is p.VAR_POSITIONAL or p.kind is p.VAR_KEYWORD:
            raise TypeError(f"variable-length parameter {name!r} is not supported")
        if p.kind is p.KEYWORD_ONLY:
            raise TypeError(f"keyword-only parameter {name!r} is not supported")
        if p.default is not inspect.Parameter.empty:
            raise TypeError(f"default value on parameter {name!r} is not supported")
        params.append(name)
    return tuple(params)


class _JittedFunction:
    """The callable produced by ``Wrapper.jit``."""

    def __init__(self, wrapper: 'Wrapper', fn: Callable, params: tuple[str, ...]) -> None:
        self._wrapper = wrapper
        self._fn = fn
        self._params = params
        self._names = {name: symbol(name) for name in params}
        self._assigns: list[AssignExpr] | None = None
        self._used_symbols: set[Symbol] | None = None
        self._cache: dict[tuple[tuple[LowerType, int], ...], CompiledWrapper] = {}
        self.__name__ = getattr(fn, '__name__', 'jitted')
        self.__doc__ = getattr(fn, '__doc__', None)

    def _trace(self) -> tuple[list[AssignExpr], set[Symbol]]:
        if self._assigns is None:
            trace = _Trace(self._names)
            probes = [trace.make_param_probe(name) for name in self._params]
            result = self._fn(*probes)
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
            self._assigns = trace.assigns
            self._used_symbols = used
        assert self._assigns is not None and self._used_symbols is not None
        return self._assigns, self._used_symbols

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

    def _compile(self, signature: tuple[tuple[LowerType, int], ...]) -> CompiledWrapper:
        assigns, _ = self._trace()
        context = TypeContext()
        for name, (lower_type, dim) in zip(self._params, signature):
            context.set_symbol(self._names[name], lower_type, dim)
        compiler = JitCompiler(
            self._wrapper._backend,
            real_type=self._wrapper._real_type,
            index_type=self._wrapper._index_type,
        )
        return compiler.compile_assignments(assigns, context)

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise TypeError(f"{self.__name__}() does not support keyword arguments")
        if len(args) != len(self._params):
            raise TypeError(
                f"{self.__name__}() takes {len(self._params)} positional arguments but {len(args)} were given"
            )
        signature = tuple(self._infer_arg_type(arg) for arg in args)
        compiled = self._cache.get(signature)
        if compiled is None:
            compiled = self._compile(signature)
            self._cache[signature] = compiled
        _, used = self._trace()
        arg_map = {self._names[n]: v for n, v in zip(self._params, args) if self._names[n] in used}
        return compiled.call(arg_map)

    def print_all(self):
        """Print the LLVM IR of the most recently compiled kernel."""
        if len(self._cache) == 0:
            return []
        return list(self._cache.values())[-1].print_all()

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

    def jit(self, fn: Callable | None = None):
        """Decorator that compiles a function of element-wise array assignments."""
        def decorator(f: Callable) -> _JittedFunction:
            return _JittedFunction(self, f, _infer_params(f))
        if fn is not None:
            return decorator(fn)
        return decorator
