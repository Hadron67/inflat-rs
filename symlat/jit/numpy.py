"""A numpy-style frontend that JIT-compiles element-wise array expressions.

Unlike :mod:`symlat.jit.fn_wrapper`, which traces the body of a decorated
function with probe objects, this module builds the expression tree directly
from the array operators: ``a + b`` only records a symbolic ``Plus`` node and
does no work.  The computation is deferred until the result is assigned into a
concrete array (``d[...] = c`` or a slice like ``a[0] = c``), which compiles
the tree into a kernel (cached per assignment) and runs it with the arrays as
arguments.
"""

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from typing_extensions import override

from ..expr import (
    AssignExpr,
    Cos,
    Exp,
    Expr,
    Flip,
    Int,
    Ln,
    Power,
    Rational,
    Roll,
    Sin,
    Slice,
    Symbol,
    Times,
    next_head_sort_token,
)
from .backend import Backend
from .compile import CompiledWrapper, JitCompiler, StandardLayoutMode
from .type import LowerType, SymbolTypeDesc


def _as_expr(value) -> Expr:
    """Convert an array wrapper or a Python/numpy scalar into an ``Expr``."""
    if isinstance(value, ArrayWrapper):
        return value.arr
    if isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.floating):
        value = float(value)
    try:
        return Expr.as_expr(value)
    except ValueError as e:
        raise TypeError(
            f"unsupported operand of type {type(value).__name__} in lazy array expression"
        ) from e


def _collect_array_nodes(expr: Expr) -> list['ArrayNode']:
    """The unique :class:`ArrayNode` leaves of an expression, in first-seen order."""
    todo = [expr]
    ret: list[ArrayNode] = []
    seen: set[ArrayNode] = set()
    while todo:
        e = todo.pop()
        if isinstance(e, ArrayNode):
            if e not in seen:
                seen.add(e)
                ret.append(e)
        else:
            todo.extend(e.subexpressions())
    return ret


def _substitute_array_nodes(expr: Expr, symbols: dict['ArrayNode', Symbol]) -> Expr:
    """Replace every :class:`ArrayNode` leaf with its kernel symbol.

    ``Expr.map`` applies its operator to every node of the tree, so a single map
    pass reaches the ``ArrayNode`` leaves at any depth.
    """
    def subst(e: Expr) -> Expr:
        if isinstance(e, ArrayNode):
            return symbols[e]
        return e
    return expr.map(subst)


def _determine_layout(values) -> StandardLayoutMode:
    """Pick the kernel layout that matches the actual array arguments.

    Returns ``ROW_MAJOR`` when every array is C-contiguous, ``COLUMN_MAJOR`` when
    every array is F-contiguous, and ``NONE`` (generic kernel) otherwise."""
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


#: numpy unary ufuncs supported as lazy element-wise functions
_FUNC_MAP = {
    'sin': Sin,
    'cos': Cos,
    'exp': Exp,
    'log': Ln,
}


@dataclass(frozen=True)
class _AssignmentCacheKey:
    """Cache key of a compiled assignment kernel.

    A kernel is fully determined by the (substituted) assignment structure, the
    dtypes and ranks of its array arguments and the layout mode; the concrete
    shapes are runtime arguments and do not participate.
    """
    lhs: Expr
    rhs: Expr
    input_types: tuple[tuple[str, int], ...]
    dest_type: tuple[str, int]
    layout: StandardLayoutMode


@dataclass(frozen=True)
class _ReductionCacheKey:
    """Cache key of a compiled reduction kernel, analogous to
    :class:`_AssignmentCacheKey` for ``sum``."""
    expr: Expr
    input_types: tuple[tuple[str, int], ...]
    layout: StandardLayoutMode


class JitContext:
    """
    A JIT context for numpy expressions.

    Offers multi-backend support and JIT cache.

    Usage example:

        np = JitContext(backend)

        a = np.rand(8, 9, 10)
        b = np.rand(8, 9, 10)

        c = a + b # computations are lazy: this only creats a symbolic expression `a + b` and does not compute the result
        d = np.zeros(*a.shape)
        d[...] = c # this triggers the computation of `c` and stores the result in `d` (d[:] = c works too)

        s = np.sum(a) # eager: reduces `a` over all axes to a scalar right away
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._compiler = JitCompiler(backend)
        # assignment structure + input/dest dtypes and ranks -> compiled kernel
        self._cache: dict[_AssignmentCacheKey, CompiledWrapper] = {}
        # reduction structure + input dtypes and ranks -> compiled kernel
        self._reduction_cache: dict[_ReductionCacheKey, CompiledWrapper] = {}

    def rand(self, *shape) -> 'ArrayWrapper':
        """A random array with entries uniformly distributed in ``[0, 1)``."""
        return ArrayWrapper(self, ArrayNode(np.random.rand(*shape)))

    def zeros(self, *shape) -> 'ArrayWrapper':
        """A zero-filled array; typically the destination of an assignment."""
        return ArrayWrapper(self, ArrayNode(np.zeros(shape)))

    def _execute(self, lhs: Expr, value) -> None:
        """Compile ``lhs = value`` (cached per assignment) and run it.

        ``lhs`` is the whole ``ArrayNode`` of the destination or a ``Slice``
        over it (a slice assignment like ``a[0] = ...``).
        """
        # the assignment target is (a slice over) a concrete array: check the
        # base before normalizing, so that assigning into a computed expression
        # reports a clear error instead of tripping over its uncomparable nodes
        base = lhs
        while isinstance(base, Slice):
            base = base.expr
        if not isinstance(base, ArrayNode):
            raise TypeError(
                "cannot assign into a computed expression; assign into an array "
                "created by rand() or zeros() instead"
            )
        lhs = lhs.normalize()
        base_arr = base.arr
        rhs = _as_expr(value)
        inputs = _collect_array_nodes(rhs)
        # every ArrayNode leaf is replaced by a fresh symbol before compiling;
        # the names are positional so that structurally identical assignments
        # (with the same input dtypes and ranks) share the cached kernel
        symbols = {node: Symbol(('@array', str(i))) for i, node in enumerate(inputs)}
        if base in symbols:
            # in-place update: the destination is also read from
            dest_sym = symbols[base]
        else:
            dest_sym = Symbol(('@dest',))
            symbols[base] = dest_sym
        # slice axis/index bounds and broadcasting compatibility are validated
        # by the compiled kernel at call time (see ``CompiledWrapper.call``)
        input_args = [
            (symbols[node], SymbolTypeDesc(LowerType.from_numpy_dtype(str(node.arr.dtype)), node.arr.ndim))
            for node in inputs
            if node is not base
        ]
        dest_desc = SymbolTypeDesc(LowerType.from_numpy_dtype(str(base_arr.dtype)), base_arr.ndim)
        args = input_args + [(dest_sym, dest_desc)]
        assign = AssignExpr(
            _substitute_array_nodes(lhs, symbols),
            _substitute_array_nodes(rhs, symbols).normalize(),
        )
        layout = _determine_layout([node.arr for node in inputs if node is not base] + [base_arr])
        key = _AssignmentCacheKey(
            assign.lhs,
            assign.rhs,
            tuple((str(node.arr.dtype), node.arr.ndim) for node in inputs if node is not base),
            (str(base_arr.dtype), base_arr.ndim),
            layout,
        )
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compiler.compile_assignments(args, [assign], standard_layout=layout)
            self._cache[key] = compiled
        compiled.call(*([node.arr for node in inputs if node is not base] + [base_arr]))

    def sum(self, array, axis=None, **kwargs) -> Any:
        """Sum ``array`` over all axes and return the scalar immediately.

        Unlike the element-wise operators, ``sum`` is eager: the reduction
        kernel is compiled (and cached per expression structure) and run right
        away, so the result is a plain numpy scalar instead of an
        :class:`ArrayWrapper`.
        """
        if axis is not None:
            raise TypeError("only np.sum over all axes is supported")
        unsupported = [
            name for name, value in kwargs.items()
            if name != 'axis' and value is not None and not isinstance(value, bool)
        ]
        if unsupported:
            raise TypeError(f"unsupported np.sum argument(s): {unsupported}")
        expr = _as_expr(array)
        inputs = _collect_array_nodes(expr)
        # slice index bounds and broadcasting compatibility are validated by the
        # compiled kernel at call time (see ``CompiledWrapper.call``)
        symbols = {node: Symbol(('@array', str(i))) for i, node in enumerate(inputs)}
        reduction = _substitute_array_nodes(expr, symbols).normalize()
        layout = _determine_layout([node.arr for node in inputs])
        key = _ReductionCacheKey(
            reduction,
            tuple((str(node.arr.dtype), node.arr.ndim) for node in inputs),
            layout,
        )
        compiled = self._reduction_cache.get(key)
        if compiled is None:
            args = [
                (symbols[node], SymbolTypeDesc(LowerType.from_numpy_dtype(str(node.arr.dtype)), node.arr.ndim))
                for node in inputs
            ]
            compiled = self._compiler.compile_reduction(args, reduction, standard_layout=layout)
            self._reduction_cache[key] = compiled
        result = compiled.call(*[node.arr for node in inputs])
        return np.asarray(result)[()]


class ArrayNode(Expr):
    """A leaf of a lazy array expression: a concrete numpy array.

    ``ArrayNode`` is not comparable: it must be replaced by a :class:`Symbol`
    (see :func:`_substitute_array_nodes`) before the expression is normalized
    or compiled.
    """

    #: a sort token distinct from every ``@exprclass``-generated class
    HEAD_SORT_TOKEN = next_head_sort_token()

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    @override
    def input_form(self) -> str:
        return f"@array{id(self)}"

    @override
    def head_sort_token(self) -> int:
        return self.HEAD_SORT_TOKEN

    @override
    def compare(self, other: Expr) -> int:
        raise NotImplementedError(
            "ArrayNode does not support comparison; replace it with a Symbol before "
            "normalizing or compiling the expression"
        )


class ArrayWrapper:
    """A lazily computed array value bound to a :class:`JitContext`.

    Arithmetic builds an expression tree without computing anything; the
    computation is triggered by assigning the value into a concrete array
    (``d[..] = c``).
    """

    def __init__(self, ctx: JitContext, arr: Expr) -> None:
        self.ctx = ctx
        self.arr = arr  # an ArrayNode leaf or a composite expression

    @property
    def shape(self) -> tuple[int, ...]:
        if not isinstance(self.arr, ArrayNode):
            raise TypeError(
                "the shape of a computed expression is not available; assign it "
                "into an array created by rand() or zeros() to materialize it"
            )
        return self.arr.arr.shape

    @property
    def ndim(self) -> int:
        if not isinstance(self.arr, ArrayNode):
            raise TypeError(
                "the ndim of a computed expression is not available; assign it "
                "into an array created by rand() or zeros() to materialize it"
            )
        return self.arr.arr.ndim

    def _new(self, expr: Expr) -> 'ArrayWrapper':
        return ArrayWrapper(self.ctx, expr)

    # --- lazy arithmetic: builds expression trees, does not compute ---------
    def __add__(self, other):
        return self._new(self.arr + _as_expr(other))

    def __radd__(self, other):
        return self._new(_as_expr(other) + self.arr)

    def __sub__(self, other):
        return self._new(self.arr - _as_expr(other))

    def __rsub__(self, other):
        return self._new(_as_expr(other) - self.arr)

    def __mul__(self, other):
        return self._new(self.arr * _as_expr(other))

    def __rmul__(self, other):
        return self._new(_as_expr(other) * self.arr)

    def __truediv__(self, other):
        return self._new(self.arr / _as_expr(other))

    def __rtruediv__(self, other):
        return self._new(_as_expr(other) / self.arr)

    def __pow__(self, other):
        return self._new(self.arr ** _as_expr(other))

    def __rpow__(self, other):
        return self._new(_as_expr(other) ** self.arr)

    def __neg__(self):
        return self._new(Times((Int(-1), self.arr)))

    def __pos__(self):
        return self

    # --- numpy functions: lazy roll/flip and scalar ufuncs -----------------
    def roll(self, shift, axis=None) -> 'ArrayWrapper':
        """Lazily roll the array along ``axis`` (or the given axes); nothing is
        computed until the result is assigned into a concrete array."""
        if axis is None:
            raise TypeError("roll requires an explicit axis")
        axes = axis if isinstance(axis, (tuple, list)) else (axis,)
        shifts = shift if isinstance(shift, (tuple, list)) else (shift,) * len(axes)
        if len(axes) != len(shifts):
            raise TypeError("roll shift and axis must have the same length")
        return self._new(
            Roll(self.arr, tuple((int(ax), int(sh)) for ax, sh in zip(axes, shifts)))
        )

    def flip(self, axis=None) -> 'ArrayWrapper':
        """Lazily flip the array along ``axis`` (or the given axes); ``None``
        flips every axis."""
        if axis is None:
            return self._new(Flip(self.arr, None))
        axes = axis if isinstance(axis, (tuple, list)) else (axis,)
        return self._new(Flip(self.arr, tuple(int(ax) for ax in axes)))

    def __array_function__(self, func, types, args, kwargs):
        """Handle ``np.roll``/``np.flip`` called on a lazy array."""
        if func is np.roll and args[0] is self:
            return self.roll(*args[1:], **kwargs)
        if func is np.flip and args[0] is self:
            return self.flip(*args[1:], **kwargs)
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle unary scalar functions like ``np.sin`` applied to a lazy array."""
        if method != '__call__' or len(kwargs) > 0:
            return NotImplemented
        name = ufunc.__name__
        if len(inputs) == 1:
            arg = inputs[0]
            if not isinstance(arg, ArrayWrapper):
                return NotImplemented
            if name == 'negative':
                return self._new(Times((Int(-1), arg.arr)))
            if name == 'positive':
                return self._new(arg.arr)
            if name == 'sqrt':
                return self._new(Power(arg.arr, Rational(1, 2)))
            fn = _FUNC_MAP.get(name)
            if fn is None:
                raise TypeError(
                    f"unsupported numpy function {name!r} in lazy array expression"
                )
            return self._new(fn(arg.arr))
        return NotImplemented

    # --- indexing: slicing is lazy, assignment triggers compilation ---------
    def _index(self, key) -> Expr:
        """The expression selected by ``self[key]``: the ``ArrayNode`` itself when
        the key selects the whole array, otherwise a ``Slice`` with one entry per
        fixed axis."""
        if key is Ellipsis:
            return self.arr
        if not isinstance(key, tuple):
            key = (key,)
        fixed: list[tuple[int, int]] = []
        for axis, k in enumerate(key):
            if k == slice(None):
                continue
            if k is Ellipsis:
                raise TypeError("'...' is not supported as an index; use explicit ':'")
            if isinstance(k, (int, np.integer)):
                fixed.append((axis, operator.index(k)))
                continue
            raise TypeError(f"unsupported index {k!r}; use integer indices or ':'")
        if len(fixed) == 0:
            return self.arr
        return Slice(self.arr, tuple(fixed))

    def __getitem__(self, key) -> 'ArrayWrapper':
        return self._new(self._index(key))

    def __setitem__(self, key, value) -> None:
        self.ctx._execute(self._index(key), value)

    def __repr__(self) -> str:
        return self.arr.input_form()
