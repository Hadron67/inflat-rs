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

import numpy as np
from typing_extensions import override

from ..expr import AssignExpr, Expr, Int, Slice, Symbol, Times
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
    """

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._compiler = JitCompiler(backend)
        # cache: (lhs, rhs, input dtypes, dest dtype, layout) -> compiled kernel
        self._cache: dict[tuple, CompiledWrapper] = {}

    def rand(self, *shape) -> 'ArrayWrapper':
        """A random array with entries uniformly distributed in ``[0, 1)``."""
        return ArrayWrapper(self, ArrayNode(np.random.rand(*shape)))

    def zeros(self, *shape) -> 'ArrayWrapper':
        """A zero-filled array; typically the destination of an assignment."""
        return ArrayWrapper(self, ArrayNode(np.zeros(shape)))

    def _execute(self, lhs: Expr, value) -> None:
        """Compile ``lhs = value`` (cached per assignment) and run it.

        ``lhs`` is the whole ``ArrayNode`` of the destination or a chain of
        ``Slice`` nodes over it (a slice assignment like ``a[0] = ...``).
        """
        # the assignment target is (a chain of slices over) a concrete array
        base = lhs
        while isinstance(base, Slice):
            base = base.expr
        if not isinstance(base, ArrayNode):
            raise TypeError(
                "cannot assign into a computed expression; assign into an array "
                "created by rand() or zeros() instead"
            )
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
        # validate the LHS slice indices
        nodes: list[Slice] = []
        cur = lhs
        while isinstance(cur, Slice):
            nodes.append(cur)
            cur = cur.expr
        remaining = list(range(base_arr.ndim))
        for node in reversed(nodes):
            if node.axis < 0 or node.axis >= len(remaining):
                raise TypeError(f"slice axis {node.axis} is out of bounds")
            axis = remaining[node.axis]
            del remaining[node.axis]
            dim = base_arr.shape[axis]
            if node.index < -dim or node.index >= dim:
                raise IndexError(
                    f"index {node.index} is out of bounds for axis {axis} of size {dim}"
                )
        dest_shape = tuple(base_arr.shape[i] for i in remaining)
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
                rem = list(range(arr.ndim))
                for node in reversed(chain):
                    if node.axis < 0 or node.axis >= len(rem):
                        raise TypeError(f"slice axis {node.axis} is out of bounds")
                    ax = rem[node.axis]
                    del rem[node.axis]
                    dim = arr.shape[ax]
                    if node.index < -dim or node.index >= dim:
                        raise IndexError(
                            f"index {node.index} is out of bounds for axis {ax} of size {dim}"
                        )
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
        the key selects the whole array, otherwise a chain of ``Slice`` nodes."""
        if key is Ellipsis:
            return self.arr
        if not isinstance(key, tuple):
            key = (key,)
        expr: Expr = self.arr
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
        # fix the highest axis first (innermost), so that lower axes stay valid
        # after the higher ones have been removed
        for axis, index in reversed(fixed):
            expr = Slice(expr, axis, index)
        return expr

    def __getitem__(self, key) -> 'ArrayWrapper':
        return self._new(self._index(key))

    def __setitem__(self, key, value) -> None:
        self.ctx._execute(self._index(key), value)

    def __repr__(self) -> str:
        return self.arr.input_form()
