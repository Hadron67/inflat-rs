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
from typing import Any

import numpy as np
from typing_extensions import override

from ..expr import AssignExpr, Expr, Int, Slice, Symbol, Times, _nth_axis
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


def _merged_loop_shape(expr: Expr) -> tuple[int, ...]:
    """The merged (broadcast) shape of every array leaf of ``expr``, in natural
    axis order -- the shape a reduction over ``expr`` iterates.

    Slice indices are validated along the way, and the leaves must be mutually
    broadcastable (numpy-style size-1 broadcasting is not supported);
    mismatches raise ``ValueError``.  The per-leaf check mirrors the assignment
    path: a sliced array whose base has the loop's rank keeps its surviving axes
    at their own positions, otherwise they are trailing-aligned with the loop.
    """
    merged: list[int] = []  # trailing-first accumulation
    leaves: list[tuple[np.ndarray, list[int]]] = []
    todo = [(expr, [])]
    while todo:
        e, chain = todo.pop()
        if isinstance(e, Slice):
            todo.append((e.expr, chain + [e]))
        elif isinstance(e, ArrayNode):
            arr = e.arr
            fixed: set[int] = set()
            for slice_node in reversed(chain):
                for k, index in slice_node.axes:
                    axis = _nth_axis(k, fixed)
                    if axis >= arr.ndim:
                        raise TypeError(f"slice axis {axis} is out of bounds")
                    dim = arr.shape[axis]
                    if index < -dim or index >= dim:
                        raise IndexError(
                            f"index {index} is out of bounds for axis {axis} of size {dim}"
                        )
                    fixed.add(axis)
            rem = [i for i in range(arr.ndim) if i not in fixed]
            for j, d in enumerate(arr.shape[ax] for ax in reversed(rem)):
                if j < len(merged):
                    if merged[j] != d:
                        raise ValueError(
                            f"cannot broadcast shape {arr.shape} with the other "
                            "summands of the reduction"
                        )
                else:
                    merged.append(d)
            leaves.append((arr, rem))
        else:
            todo.extend((c, chain) for c in e.subexpressions())
    loop_shape = tuple(reversed(merged))
    rank = len(loop_shape)
    for arr, rem in leaves:
        if len(rem) > rank:
            raise ValueError(
                f"cannot broadcast shape {arr.shape} into a reduction of shape {loop_shape}"
            )
        offset = 0 if arr.ndim == rank else rank - len(rem)
        for j, ax in enumerate(rem):
            if arr.shape[ax] != loop_shape[offset + j]:
                raise ValueError(
                    f"cannot broadcast shape {arr.shape} into a reduction of shape {loop_shape}"
                )
    return loop_shape


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
        # cache: (lhs, rhs, input dtypes, dest dtype, layout) -> compiled kernel
        self._cache: dict[tuple, CompiledWrapper] = {}
        # reduction cache: (expr, input dtypes/ranks, layout) -> compiled kernel
        self._reduction_cache: dict[tuple, CompiledWrapper] = {}

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
        # the kernel iterates the destination's (sliced) shape: compute it and
        # validate the LHS slice indices (the axes of a slice are directly
        # relative to its expression)
        if isinstance(lhs, Slice):
            fixed: set[int] = set()
            for axis, index in lhs.axes:
                if axis < 0 or axis >= base_arr.ndim:
                    raise TypeError(f"slice axis {axis} is out of bounds")
                dim = base_arr.shape[axis]
                if index < -dim or index >= dim:
                    raise IndexError(
                        f"index {index} is out of bounds for axis {axis} of size {dim}"
                    )
                fixed.add(axis)
            dest_shape = tuple(s for i, s in enumerate(base_arr.shape) if i not in fixed)
        else:
            dest_shape = base_arr.shape
        # every input must read within the loop bounds: slice indices are
        # checked, and the surviving axes must equal the destination axis they
        # align to (numpy-style size-1 broadcasting is not supported)
        todo = [(rhs, [])]
        while todo:
            expr, chain = todo.pop()
            if isinstance(expr, Slice):
                todo.append((expr.expr, chain + [expr]))
            elif isinstance(expr, ArrayNode):
                arr = expr.arr
                # map the axes of the enclosing slices to the node's numbering:
                # the axes of an outer slice are relative to the axes that
                # survive the inner slices
                fixed_axes: set[int] = set()
                for slice_node in reversed(chain):
                    for k, index in slice_node.axes:
                        axis = _nth_axis(k, fixed_axes)
                        if axis >= arr.ndim:
                            raise TypeError(f"slice axis {axis} is out of bounds")
                        dim = arr.shape[axis]
                        if index < -dim or index >= dim:
                            raise IndexError(
                                f"index {index} is out of bounds for axis {axis} of size {dim}"
                            )
                        fixed_axes.add(axis)
                rem = [i for i in range(arr.ndim) if i not in fixed_axes]
                if len(rem) > len(dest_shape):
                    raise ValueError(
                        f"cannot broadcast shape {arr.shape} into destination shape {dest_shape}"
                    )
                # a sliced expression whose base has the loop's rank is read at
                # its surviving axes positionally; otherwise (broadcasting) its
                # surviving axes are trailing-aligned with the loop
                offset = 0 if arr.ndim == len(dest_shape) else len(dest_shape) - len(rem)
                for j, ax in enumerate(rem):
                    if arr.shape[ax] != dest_shape[offset + j]:
                        raise ValueError(
                            f"cannot broadcast shape {arr.shape} into destination shape {dest_shape}"
                        )
            else:
                todo.extend((c, chain) for c in expr.subexpressions())
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
        key = (
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
        # validate slice indices and broadcasting compatibility; the compiler
        # derives the loop shape itself from the symbol dimensions
        _merged_loop_shape(expr)
        symbols = {node: Symbol(('@array', str(i))) for i, node in enumerate(inputs)}
        reduction = _substitute_array_nodes(expr, symbols).normalize()
        layout = _determine_layout([node.arr for node in inputs])
        key = (
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
    HEAD_SORT_TOKEN = 0x10000

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
