"""A numpy-style frontend that JIT-compiles element-wise array expressions.

Unlike :mod:`symlat.jit.fn_wrapper`, which traces the body of a decorated
function with probe objects, this module builds the expression tree directly
from the array operators: ``a + b`` only records a symbolic ``Plus`` node and
does no work.  The computation is deferred until the result is assigned into a
concrete array (``d[..] = c``), which compiles the tree into a kernel (cached
per assignment) and runs it with the arrays as arguments.
"""

import numpy as np
from typing_extensions import override

from ..expr import AssignExpr, Expr, Int, Symbol, Times
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


def _substitute_array_nodes(expr: Expr) -> Expr:
    """Replace every :class:`ArrayNode` leaf with its kernel symbol.

    ``Expr.map`` applies its operator to every node of the tree, so a single map
    pass reaches the ``ArrayNode`` leaves at any depth.
    """
    def subst(e: Expr) -> Expr:
        if isinstance(e, ArrayNode):
            return e.sym
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

    def _execute(self, dest: 'ArrayWrapper', rhs: Expr) -> None:
        """Compile ``dest[..] = rhs`` (cached per assignment) and run it."""
        if not isinstance(dest.arr, ArrayNode):
            raise TypeError(
                "cannot assign into a computed expression; assign into an array "
                "created by rand() or zeros() instead"
            )
        dest_arr = dest.arr.arr
        inputs = _collect_array_nodes(rhs)
        # the kernel iterates the destination's shape and reads the inputs with
        # trailing-aligned subscripts, so every input axis must equal the
        # destination axis it aligns to (numpy-style size-1 broadcasting is not
        # supported)
        dest_shape = dest_arr.shape
        for node in inputs:
            src_shape = node.arr.shape
            if len(src_shape) > len(dest_shape):
                raise ValueError(
                    f"cannot broadcast shape {src_shape} into destination shape {dest_shape}"
                )
            for d, s in zip(reversed(dest_shape), reversed(src_shape)):
                if d != s:
                    raise ValueError(
                        f"cannot broadcast shape {src_shape} into destination shape {dest_shape}"
                    )
        input_args = [
            (node.sym, SymbolTypeDesc(LowerType.from_numpy_dtype(str(node.arr.dtype)), node.arr.ndim))
            for node in inputs
            if node is not dest.arr
        ]
        dest_desc = SymbolTypeDesc(LowerType.from_numpy_dtype(str(dest_arr.dtype)), dest_arr.ndim)
        args = input_args + [(dest.arr.sym, dest_desc)]
        assign = AssignExpr(dest.arr.sym, _substitute_array_nodes(rhs).normalize())
        layout = _determine_layout([node.arr for node in inputs if node is not dest.arr] + [dest_arr])
        key = (
            assign.lhs,
            assign.rhs,
            tuple(str(node.arr.dtype) for node in inputs if node is not dest.arr),
            str(dest_arr.dtype),
            layout,
        )
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compiler.compile_assignments(args, [assign], standard_layout=layout)
            self._cache[key] = compiled
        compiled.call(*([node.arr for node in inputs if node is not dest.arr] + [dest_arr]))


class ArrayNode(Expr):
    """A leaf of a lazy array expression: a concrete numpy array.

    Each node owns a stable :class:`Symbol` that names its kernel argument, so
    the same node always maps to the same kernel slot; this is what makes
    repeated executions of an assignment hit the JIT cache.
    """

    #: a sort token distinct from every ``@exprclass``-generated class
    HEAD_SORT_TOKEN = 0x10000

    _counter = 0

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self._id = ArrayNode._counter
        ArrayNode._counter += 1
        self.sym = Symbol(('@array', str(self._id)))

    @override
    def input_form(self) -> str:
        return f"@array{self._id}"

    @override
    def head_sort_token(self) -> int:
        return self.HEAD_SORT_TOKEN

    @override
    def compare(self, other: Expr) -> int:
        if isinstance(other, ArrayNode):
            return self._id - other._id
        return self.HEAD_SORT_TOKEN - other.HEAD_SORT_TOKEN


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

    # --- assignment: triggers compilation and execution ---------------------
    def __setitem__(self, key, value) -> None:
        if key is not Ellipsis and key != slice(None):
            raise TypeError(
                "only whole-array assignment (d[..] = ... or d[:] = ...) is supported"
            )
        self.ctx._execute(self, _as_expr(value))

    def __repr__(self) -> str:
        return self.arr.input_form()
