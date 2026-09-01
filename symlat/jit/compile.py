import ctypes
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, override

import numpy as np

from ..expr import (
    AssignExpr,
    Complex,
    Coord,
    Cos,
    Exp,
    Expr,
    Flip,
    Float,
    Int,
    Ln,
    Plus,
    Power,
    Rational,
    Roll,
    Sin,
    Slice,
    Symbol,
    Times,
)
from . import type as ap
from .backend import (
    Backend,
    CompiledBackendFunction,
    DebugInterface,
    LoopKernel,
    ReductionKernel,
)
from .helper import CompileHelper, ComplexValue, MaybeComplexValue
from .llvm import (
    Add,
    BasicBlock,
    FAdd,
    FloatType,
    FloatValue,
    IntType,
    IntValue,
    Ordering,
    Value,
    VoidValue,
)
from .type import (
    ComplexFloatType,
    LowerType,
    SymbolShape,
    TypedAssignExpr,
    TypeResolver,
    TypesConfig,
    get_peer_types,
)


class SymbolArgInfo:
    @abstractmethod
    def write_one_arg(self, arg_value: Any, args: list[ctypes._CDataType | None], config: TypesConfig):
        raise NotImplementedError

@dataclass
class ScalarArgInfo(SymbolArgInfo):
    value: int
    is_ref: bool = False

    @override
    def __str__(self) -> str:
        if self.is_ref:
            return f"%{self.value}: ScalarRef"
        return f"%{self.value}: Scalar"

@dataclass
class ArrayArgInfo(SymbolArgInfo):
    ptr: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    @override
    def __str__(self) -> str:
        return f"%{self.ptr}: Array(strides=({", ".join(str(i) for i in self.strides)}))"


def _check_and_get_total_size(exprs: list[TypedAssignExpr], reduction: 'TypedReductionExpr | None' = None, resolver: 'TypeResolver | None' = None):
    assert len(exprs) > 0 or reduction is not None, "no expressions to compile"
    if len(exprs) > 0:
        first_size = exprs[0].total_size()
        rest = exprs[1:]
    else:
        assert reduction is not None
        first_size = reduction.total_size()
        rest = []

    def resolve_size(expr: Expr) -> tuple[Expr, ...]:
        if resolver is None:
            resolved = expr
        else:
            resolved_shapes = resolver.resolved_shapes
            resolved = expr.map(lambda e: resolved_shapes.get(e, e) if isinstance(e, SymbolShape) else e)
        # total sizes are products of shape variables; compare them as multisets,
        # since the shape variables may have been constrained to equal each other
        # in different orders
        if isinstance(resolved, Times):
            return tuple(sorted(resolved.children))
        return (resolved,)

    first_resolved = resolve_size(first_size)
    for expr in rest:
        expr_size = expr.total_size()
        assert first_resolved == resolve_size(expr_size), f"incompatible expressions {exprs[0]} and {expr}, with incompatible total sizes {first_size} and {expr_size}"
    if reduction is not None:
        reduction_size = reduction.total_size()
        assert first_resolved == resolve_size(reduction_size), f"incompatible reduction expression {reduction.expr}, with total size {reduction_size}, expected {first_size}"

    assert first_size is not None, "cannot compile expression with unspecified shapes"

    return first_size

def _collect_lvalue_symbols(exprs: list[AssignExpr]) -> set[Symbol]:
    """The symbols written to by the assignments, i.e. the base symbols of their
    left-hand sides.

    A scalar reference argument that is never written to does not need a pointer:
    it can be passed by value, which avoids a pointer load on every read."""
    ret: set[Symbol] = set()
    for e in exprs:
        lhs = e.lhs
        while isinstance(lhs, Slice):
            lhs = lhs.expr
        if isinstance(lhs, Symbol):
            ret.add(lhs)
    return ret

class StandardLayoutMode(Enum):
    NONE = "none"
    COLUMN_MAJOR = "column"
    ROW_MAJOR = "row"


def _gen_call_invoke(symbols: '_SymbolScope', parent: 'JitCompiler', standard_layout: StandardLayoutMode) -> Callable:
    """Generate the ``__invoke(self, values)`` function of a
    :class:`CompiledWrapper`.

    The function is specialised per kernel: the argument-count check, the
    layout check, the shape validation and the per-symbol ctypes conversion are
    unrolled into straight-line code, so a call reduces to plain indexing and
    conversions instead of an interpreted loop."""
    index_ctype = parent.index_type.to_ctype()
    symbol_order = symbols.get_symbol_order()
    arg_count = len(symbol_order)
    pos_of = {symbol: i for i, symbol in enumerate(symbol_order)}

    def shape_eval(expr: Expr) -> str:
        """A Python expression evaluating ``expr`` against the ``values`` tuple
        at call time, or ``None`` when the dimension cannot be evaluated."""
        if isinstance(expr, SymbolShape):
            pos = pos_of.get(expr.symbol)
            if pos is None:
                return 'None'
            return (
                f'values[{pos}].shape[{expr.index}]'
                f' if isinstance(values[{pos}], np.ndarray)'
                f' and 0 <= {expr.index} < values[{pos}].ndim else None'
            )
        if isinstance(expr, Int):
            return str(expr.value)
        return 'None'

    globals: dict[str, Any] = {'np': np, 'ctypes': ctypes, '_INDEX': index_ctype}
    lines: list[str] = ['def __invoke(self, values):']
    lines += [
        f'    if len(values) != {arg_count}:',
        f"        raise TypeError(f\"the kernel expects {arg_count} positional argument(s), got {{len(values)}}\")",
    ]
    if standard_layout is not StandardLayoutMode.NONE:
        flag = 'C_CONTIGUOUS' if standard_layout is StandardLayoutMode.ROW_MAJOR else 'F_CONTIGUOUS'
        msg = (
            f"the kernel was compiled for {standard_layout.value} layout but the array "
            "arguments do not match; pass contiguous arrays of the expected layout or recompile "
            "with standard_layout=StandardLayoutMode.NONE"
        )
        for i in range(arg_count):
            lines += [
                f'    if isinstance(values[{i}], np.ndarray) and not values[{i}].flags[{flag!r}]:',
                f'        raise ValueError({msg!r})',
            ]
    type_cache = symbols.type_cache
    for lhs, rhs in type_cache.resolved_shapes.items():
        lines += [
            f'    v = {shape_eval(rhs)}',
            f'    s = {shape_eval(lhs)}',
            '    if v != s:',
            '        raise ValueError(f"resolved shape {v} does not match {s}")',
        ]
    for length, index in type_cache.slice_checks:
        lines += [
            f'    d = {shape_eval(length)}',
            f'    if d is not None and not -d <= {index} < d:',
            f'        raise IndexError(f"slice index {index} is out of bounds for a dimension of size {{d}}")',
        ]
    lines.append(f'    ret = [None] * {symbols.get_arg_count()}')
    for i, symbol in enumerate(symbol_order):
        info = symbols.get_symbol(symbol)
        lower_ctype = symbols.type_cache.get_symbol_type(symbol).to_ctype()
        lower_size = ctypes.sizeof(lower_ctype)
        match info:
            case ScalarArgInfo(slot, is_ref=True):
                globals[f'_PTR{i}'] = ctypes.POINTER(lower_ctype)
                lines += [
                    f'    if isinstance(values[{i}], np.ndarray):',
                    f'        ret[{slot}] = ctypes.cast(values[{i}].ctypes.data, _PTR{i})',
                    '    else:',
                    f'        ret[{slot}] = ctypes.pointer(values[{i}])',
                ]
            case ScalarArgInfo(slot, is_ref=False):
                globals[f'_CTYPE{i}'] = lower_ctype
                lines += [
                    f'    if isinstance(values[{i}], ctypes._SimpleCData):',
                    f'        ret[{slot}] = _CTYPE{i}(values[{i}].value)',
                    '    else:',
                    f'        ret[{slot}] = _CTYPE{i}(values[{i}])',
                ]
            case ArrayArgInfo(ptr, shape_slots, stride_slots):
                globals[f'_PTR{i}'] = ctypes.POINTER(lower_ctype)
                globals[f'_SIZE{i}'] = lower_size
                lines += [
                    f'    ret[{ptr}] = ctypes.cast(values[{i}].ctypes.data, _PTR{i})',
                    f'    assert len(values[{i}].shape) == {len(shape_slots)}',
                    f'    assert len(values[{i}].strides) == {len(stride_slots)}',
                ]
                for j, slot in enumerate(shape_slots):
                    lines.append(f'    ret[{slot}] = _INDEX(values[{i}].shape[{j}])')
                for j, slot in enumerate(stride_slots):
                    lines += [
                        f'    assert values[{i}].strides[{j}] % _SIZE{i} == 0',
                        f'    ret[{slot}] = _INDEX(values[{i}].strides[{j}] // _SIZE{i})',
                    ]
    lines += [
        '    assert None not in ret',
        '    return self._inner.call(*ret)',
    ]
    exec('\n'.join(lines), globals)  # noqa: S102
    return globals['__invoke']

class _SymbolScope:
    def __init__(self, type_cache: TypeResolver) -> None:
        self._symbol_values: dict[Symbol, SymbolArgInfo] = {}
        self._symbol_order: list[Symbol] = []
        self._args: list[LowerType] = []
        self.type_cache = type_cache

    def get_symbol(self, symbol: Symbol):
        return self._symbol_values[symbol]

    def get_symbol_order(self) -> tuple[Symbol, ...]:
        """The symbols in the order the compiled function takes its positional arguments."""
        return tuple(self._symbol_order)

    def get_args(self) -> tuple[LowerType, ...]:
        return tuple(self._args)

    def get_arg_count(self) -> int:
        return len(self._args)

    def _add_arg(self, type: LowerType):
        ret = len(self._args)
        self._args.append(type)
        return ret

    def add_symbol(self, symbol: Symbol, by_ref: set[Symbol] | None = None):
        """Register one function argument.  The registration order is the positional
        argument order of the compiled function."""
        if symbol in self._symbol_values:
            raise ValueError(f"duplicate symbol {symbol} in function arguments")
        lower_type = self.type_cache.get_symbol_type(symbol)
        dim = self.type_cache.get_symbol_dimension(symbol)
        is_ref = by_ref is not None and symbol in by_ref
        if dim == 0:
            if is_ref:
                # scalar references are passed as pointers so that writes
                # propagate back to the caller
                ret = ScalarArgInfo(self._add_arg(ap.PointerType(lower_type)), is_ref=True)
            else:
                # scalar arguments are passed by value
                ret = ScalarArgInfo(self._add_arg(lower_type))
        else:
            # TODO: check indices types
            ret = ArrayArgInfo(
                self._add_arg(ap.PointerType(lower_type)),
                tuple(self._add_arg(self.type_cache.type_config.index_type) for _ in range(dim)),
                tuple(self._add_arg(self.type_cache.type_config.index_type) for _ in range(dim)),
            )
        self._symbol_values[symbol] = ret
        self._symbol_order.append(symbol)

    @override
    def __str__(self):
        elems: list[str] = []
        for sym, info in self._symbol_values.items():
            match info:
                case ScalarArgInfo(is_ref=True):
                    elems.append(f"%{info.value}: ScalarRef = {sym}")
                case ScalarArgInfo():
                    elems.append(f"%{info.value}: Scalar = {sym}")
                case ArrayArgInfo():
                    strides = ', '.join(f"%{s}" for s in info.strides)
                    elems.append(f"%{info.ptr}: Array(strides=({strides})) = {sym}")
        return f"SymbolScope({', '.join(elems)})"

class _SubscriptsInfo:
    pass

@dataclass
class _RealSubscriptsInfo(_SubscriptsInfo):
    subscripts: tuple[Value, ...]

@dataclass
class _StandardLayoutSubscriptInfo(_SubscriptsInfo):
    mode: StandardLayoutMode
    subscript: Value
    shifts: dict[int, int]

class _FunctionCompiler:
    def __init__(
        self,
        parent: 'JitCompiler',
        helper: CompileHelper,
        args: tuple[Value, ...],
        block: BasicBlock,
        symbol_scope: _SymbolScope,
        debug: DebugInterface | None = None,
        standard_layout: StandardLayoutMode = StandardLayoutMode.NONE,
    ) -> None:
        self.parent = parent
        self._args = args
        self._block = block
        self._symbol_scope = symbol_scope
        self._helper = helper
        self._expr_cache: dict[tuple[Any, ...], MaybeComplexValue] = {}
        self._subscript_cache: dict[tuple[tuple[Value, ...], tuple[Value, ...]], Value] = {}
        self._finished: bool = False
        self._type_cache = symbol_scope.type_cache
        self._standard_layout = standard_layout
        self._debug = debug

    def _add(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType, result_type: LowerType) -> MaybeComplexValue:
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_add(self._block, left, right)
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.add(left, right)

    def _sub(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType, result_type: LowerType) -> MaybeComplexValue:
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_sub(self._block, left, right)
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.sub(left, right)

    def _mul(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType, result_type: LowerType) -> MaybeComplexValue:
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_mul(self._block, left, right)
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.mul(left, right)

    def _div(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType, result_type: LowerType) -> MaybeComplexValue:
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_div(self._block, left, right)
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.div(left, right, True)

    def _sqrt(self, expr: MaybeComplexValue, type: ap.LowerType) -> MaybeComplexValue:
        if isinstance(type, ComplexFloatType):
            raise NotImplementedError
        assert not isinstance(expr, ComplexValue)
        return self._block.sqrt(expr)

    def _int_pow(self, base: MaybeComplexValue, base_type: ap.LowerType, exp: int, result_type: LowerType) -> MaybeComplexValue:
        neg = False
        if exp < 0:
            exp = -exp
            neg = True
        ret = base
        for _ in range(exp - 1):
            ret = self._mul(ret, base_type, base, base_type, base_type)
        ret = self._helper.coerce(self._block, ret, base_type, result_type)
        if neg:
            ret = self._div(result_type.to_llvm_type().from_int(1), result_type, ret, result_type, result_type)
        return ret

    def _pow(self, base: MaybeComplexValue, base_type: ap.LowerType, exp: MaybeComplexValue, exp_type: ap.LowerType, result_type: LowerType) -> MaybeComplexValue:
        base = self._helper.coerce(self._block, base, base_type, result_type)
        exp = self._helper.coerce(self._block, exp, result_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            raise NotImplementedError
        assert not isinstance(base, ComplexValue) and not isinstance(exp, ComplexValue)
        return self._block.pow(base, exp)

    def _store(self, ptr: Value, value: MaybeComplexValue):
        b = self._block
        match value:
            case ComplexValue(re, im):
                b.store(b.get_element_ptr(ptr, 0, 0), re)
                b.store(b.get_element_ptr(ptr, 0, 1), im)
            case _:
                b.store(ptr, value)

    def _normalize_slice_index(self, length: Value, index: int) -> Value:
        """Normalize a (possibly negative) slice index to ``index mod length``."""
        h = self._helper
        if index >= 0:
            return IntValue(index, h.llvm_index_type)
        abs_index = IntValue(-index, h.llvm_index_type)
        # ceil(abs_index / length)
        multiples = self._block.div(
            self._block.add(abs_index, self._block.sub(length, IntValue(1, h.llvm_index_type))),
            length,
            False,
        )
        return self._block.rem(
            self._block.add(IntValue(index, h.llvm_index_type), self._block.mul(multiples, length)),
            length,
            False,
        )

    def _flat_array_index(self, symbol: Symbol, subscripts: _StandardLayoutSubscriptInfo) -> Value:
        """Compute the linear index of an array in standard layout mode: the flat
        subscript plus, for every sliced axis, the (normalized) slice index times the
        stride of that axis."""
        sym = self._symbol_scope.get_symbol(symbol)
        assert isinstance(sym, ArrayArgInfo)
        index = subscripts.subscript
        for axis, raw_index in subscripts.shifts.items():
            # record the bounds check; it is verified against the concrete
            # dimension at call time
            self._type_cache.slice_checks.append((SymbolShape(symbol, axis), raw_index))
            length = self._args[sym.shape[axis]]
            idx = self._normalize_slice_index(length, raw_index)
            stride = self._args[sym.strides[axis]]
            index = self._block.add(index, self._block.mul(idx, stride))
        return index

    def _compile_slice_chain(self, expr: Slice, subscripts: _RealSubscriptsInfo) -> MaybeComplexValue:
        """Compile a sliced expression with the loop subscripts: fix the sliced
        axes at their indices and address the base expression.

        The axes of a ``Slice`` are directly relative to its expression, so a
        nested slice (e.g. ``np.sin(b)[2]``) recurses through
        :meth:`compile_expr` with its own axes."""
        cur = expr.expr
        cur_shape = self._type_cache.get_shape(cur)
        dim = len(cur_shape)
        r = len(subscripts.subscripts)
        entries: list[tuple[int, Value]] = []
        for axis, index in expr.axes:
            if axis < 0 or axis >= dim:
                raise IndexError(f"slice axis {axis} is out of bounds for {cur}")
            length = self.compile_non_complex_expr(cur_shape[axis], _RealSubscriptsInfo(()))
            # record the bounds check; it is verified against the concrete
            # dimension at call time
            self._type_cache.slice_checks.append((cur_shape[axis], index))
            entries.append((axis, self._normalize_slice_index(length, index)))
        if dim == r:
            # the loop covers every axis of the sliced expression: keep the loop
            # subscripts and replace the value at the fixed axes
            v: list[Value] = list(subscripts.subscripts)
            for axis, index in entries:
                v[axis] = index
            return self.compile_expr(cur, _RealSubscriptsInfo(tuple(v)))
        # the loop does not cover every axis: the sliced expression is broadcast,
        # so the surviving axes (ascending) receive the trailing-aligned loop
        # subscripts and the fixed axes receive their indices
        fixed = {axis for axis, _ in entries}
        surviving = [axis for axis in range(dim) if axis not in fixed]
        r_rhs = len(surviving)
        value_at = {axis: index for axis, index in entries}
        for j, axis in enumerate(surviving):
            value_at[axis] = subscripts.subscripts[r - r_rhs + j]
        v = [value_at[axis] for axis in range(dim)]
        return self.compile_expr(cur, _RealSubscriptsInfo(tuple(v)))

    def _compile_slice_lvalue(self, expr: Slice, subscripts: _RealSubscriptsInfo) -> tuple[Value, LowerType]:
        """Compile the lvalue of a slice expression, e.g. ``a[0]`` or ``a[:, 3]``.

        Every fixed axis is addressed at its index; the surviving axes receive
        the loop subscripts in order, since the loop iterates exactly the sliced
        shape."""
        cur = expr.expr
        cur_shape = self._type_cache.get_shape(cur)
        dim = len(cur_shape)
        fixed = {axis for axis, _ in expr.axes}
        surviving = [axis for axis in range(dim) if axis not in fixed]
        if len(subscripts.subscripts) != len(surviving):
            raise TypeError("slice assignment loop rank does not match the sliced expression")
        value_at: dict[int, Value] = {}
        for axis, index in expr.axes:
            if axis < 0 or axis >= dim:
                raise IndexError(f"slice axis {axis} is out of bounds for {cur}")
            length = self.compile_non_complex_expr(cur_shape[axis], _RealSubscriptsInfo(()))
            # record the bounds check; it is verified against the concrete
            # dimension at call time
            self._type_cache.slice_checks.append((cur_shape[axis], index))
            value_at[axis] = self._normalize_slice_index(length, index)
        for j, axis in enumerate(surviving):
            value_at[axis] = subscripts.subscripts[j]
        v = [value_at[axis] for axis in range(dim)]
        return self._compile_lvalue(cur, _RealSubscriptsInfo(tuple(v)))

    def _compile_unpack_subscripts(self, sizes: tuple[Value, ...], packed: Value) -> tuple[Value, ...]:
        """Unpack the flat loop index into one subscript per axis.

        ``sizes`` is the shape in natural axis order (outermost first, like
        numpy's ``.shape``); the flat index iterates the innermost axis
        fastest, so the unpacking starts from the last axis."""
        assert len(sizes) > 0
        ret: list[Value] = []
        for size in reversed(sizes[1:]):
            ret.append(self._block.rem(packed, size, False))
            packed = self._block.div(packed, size, False)
        ret.append(packed)
        return tuple(reversed(ret))

    def _compile_subscript_no_cache(self, strides: tuple[Value, ...], subscripts: tuple[Value, ...]) -> Value:
        assert len(subscripts) >= len(strides), f"incompatible subscripts {subscripts} and strides {strides}"
        assert len(strides) > 0
        index = self._block.mul(subscripts[-1], strides[-1])
        for i in range(1, min(len(strides), len(subscripts))):
            index = self._block.add(index, self._block.mul(subscripts[-1 - i], strides[-1 - i]))
        return index

    def _compile_subscript(self, strides: tuple[Value, ...], subscripts: tuple[Value, ...]) -> Value:
        cache_key = (subscripts, strides)
        if cache_key in self._subscript_cache:
            return self._subscript_cache[cache_key]
        index = self._compile_subscript_no_cache(strides, subscripts)
        self._subscript_cache[cache_key] = index
        return index

    def _compile_array_symbol_access(self, info: ArrayArgInfo, subscripts: tuple[Value, ...]) -> Value:
        index = self._compile_subscript(tuple(self._args[i] for i in info.strides), subscripts)
        return self._block.get_element_ptr(
            self._args[info.ptr],
            index,
        )

    def _from_lower_real_value(self, value: Value, lower_type: LowerType):
        return self._helper.coerce_lower_type(self._block, value, lower_type, self.parent.real_type)

    def _echo(self, *args: tuple[MaybeComplexValue, ap.LowerType] | str):
        if self._debug is not None:
            converted_args: list[Value | str] = []
            for arg in args:
                if isinstance(arg, tuple):
                    if isinstance(arg[1], ap.ComplexFloatType):
                        re, im = self._helper.expand_complex_value(self._block, arg[0])
                        converted_args.extend(['complex(', re, ', ', im, ')'])
                    else:
                        assert not isinstance(arg[0], ComplexValue)
                        converted_args.append(arg[0])
                else:
                    converted_args.append(arg)
            self._debug.echo(self._block, *converted_args)

    def _compile_expr_no_cache(self, expr: Expr, subscripts: _SubscriptsInfo) -> MaybeComplexValue:
        h = self._helper

        expr_type = self._type_cache.get_type(expr)
        match expr:
            case Int(value):
                return IntValue(value, h.llvm_index_type)
            case Rational(numerator, denominator):
                return FloatValue(numerator / denominator, h.llvm_real_type)
            case Float(value):
                return FloatValue(value, h.llvm_real_type)
            case Complex(re, im):
                assert isinstance(expr_type, ap.ComplexFloatType)
                re_value = self.compile_non_complex_expr(re, subscripts)
                im_value = self.compile_non_complex_expr(im, subscripts)
                re_type = self._type_cache.get_type(re)
                im_type = self._type_cache.get_type(im)
                re_value = self._helper.coerce(self._block, re_value, re_type, expr_type.type)
                im_value = self._helper.coerce(self._block, im_value, im_type, expr_type.type)
                assert not isinstance(re_value, ComplexValue) and not isinstance(im_value, ComplexValue)
                return ComplexValue(re_value, im_value)
            case Symbol():
                sym = self._symbol_scope.get_symbol(expr)
                lower_type = self._type_cache.get_symbol_type(expr)
                ret = None
                match sym:
                    case ScalarArgInfo(is_ref=True):
                        # scalar references are passed as pointers; dereference on read
                        ret = self._block.load(self._args[sym.value])
                    case ScalarArgInfo():
                        ret = self._args[sym.value]
                    case ArrayArgInfo():
                        match subscripts:
                            case _RealSubscriptsInfo(s):
                                ret = self._block.load(self._compile_array_symbol_access(sym, s))
                            case _StandardLayoutSubscriptInfo():
                                # standard layout: the loop variable is the linear array index
                                ret = self._block.load(
                                    self._block.get_element_ptr(self._args[sym.ptr], self._flat_array_index(expr, subscripts))
                                )
                    case _:
                        raise NotImplementedError
                assert ret is not None
                return self._helper.coerce(self._block, ret, lower_type, expr_type)
            case SymbolShape(symbol, index):
                sym = self._symbol_scope.get_symbol(symbol)
                assert isinstance(sym, ArrayArgInfo), "SymbolShape must be used with an array symbol"
                assert index < len(sym.shape), "SymbolShape index out of bounds"
                return self._args[sym.shape[index]]
            case Coord(axis, _):
                # the coordinate is the loop subscript along the given axis
                match subscripts:
                    case _RealSubscriptsInfo(s):
                        if axis < 0 or axis >= len(s):
                            raise IndexError(
                                f"coord axis {axis} is out of bounds for a {len(s)}-dimensional loop"
                            )
                        return s[axis]
                    case _StandardLayoutSubscriptInfo():
                        raise TypeError("coord is not supported in standard layout kernels")
            case Roll():
                assert isinstance(subscripts, _RealSubscriptsInfo), "cannot compile Roll in standard layout mode"
                expr_shape = self._type_cache.get_shape(expr.expr)
                assert expr_shape is not None, "cannot compile unspecified shape"
                axes = expr.axes
                if isinstance(axes, int):
                    # rolling every axis by the same amount: the rank is only
                    # known here
                    axes = tuple((i, axes) for i in range(len(expr_shape)))
                # the rolled expression may have a lower rank than the loop
                # (broadcasting); its axes are trailing-aligned with the loop
                # subscripts, like array access
                sub = subscripts.subscripts
                for axis, amount in axes:
                    if axis < 0:
                        axis += len(expr_shape)
                    subscript_index = len(sub) - len(expr_shape) + axis
                    if subscript_index < 0 or subscript_index >= len(sub):
                        raise IndexError(
                            f"np.roll axis {axis} is out of bounds for the loop"
                        )
                    length = self.compile_non_complex_expr(expr_shape[axis], _RealSubscriptsInfo(()))
                    length_type = self._type_cache.get_type(expr_shape[axis])
                    assert isinstance(length_type, ap.IntType), "length must be an integer"
                    # new index = (subscript - amount) mod length; add enough
                    # multiples of the length so that the unsigned remainder is
                    # well-defined
                    abs_amount = IntValue(abs(amount), h.llvm_index_type)
                    # ceil(abs_amount / length)
                    multiples = self._block.div(
                        self._block.add(abs_amount, self._block.sub(length, IntValue(1, h.llvm_index_type))),
                        length,
                        False,
                    )
                    new_index = self._block.add(
                        self._block.add(sub[subscript_index], IntValue(-amount, h.llvm_index_type)),
                        self._block.mul(multiples, length),
                    )
                    new_index = self._block.rem(new_index, length, False)
                    sub = sub[:subscript_index] + (new_index,) + sub[subscript_index + 1:]
                return self.compile_expr(expr.expr, _RealSubscriptsInfo(tuple(sub)))
            case Flip():
                assert isinstance(subscripts, _RealSubscriptsInfo), "cannot compile Flip in standard layout mode"
                expr_shape = self._type_cache.get_shape(expr.expr)
                assert expr_shape is not None, "cannot compile unspecified shape"
                axes = expr.axes
                if axes is None:
                    # flipping every axis: the rank is only known here
                    axes = tuple(range(len(expr_shape)))
                # the flipped expression may have a lower rank than the loop
                # (broadcasting); its axes are trailing-aligned with the loop
                # subscripts, like array access
                sub = subscripts.subscripts
                for axis in axes:
                    if axis < 0:
                        axis += len(expr_shape)
                    length = self.compile_non_complex_expr(expr_shape[axis], _RealSubscriptsInfo(()))
                    subscript_index = len(sub) - len(expr_shape) + axis
                    if subscript_index < 0 or subscript_index >= len(sub):
                        raise IndexError(
                            f"np.flip axis {axis} is out of bounds for the loop"
                        )
                    # new index = length - 1 - subscript
                    new_index = self._block.sub(
                        self._block.sub(length, IntValue(1, h.llvm_index_type)),
                        sub[subscript_index],
                    )
                    sub = sub[:subscript_index] + (new_index,) + sub[subscript_index + 1:]
                return self.compile_expr(expr.expr, _RealSubscriptsInfo(tuple(sub)))
            case Slice():
                match subscripts:
                    case _StandardLayoutSubscriptInfo(mode, subscript, shifts):
                        # record the fixed axes and recurse; the offset is applied
                        # at the array access using the base array's strides
                        new_shifts = dict(shifts)
                        rank = len(self._type_cache.get_shape(expr.expr))
                        for axis, index in expr.axes:
                            if axis < 0 or axis >= rank:
                                raise IndexError(
                                    f"slice axis {axis} is out of bounds for {expr.expr}"
                                )
                            new_shifts[axis] = index
                        return self.compile_expr(
                            expr.expr, _StandardLayoutSubscriptInfo(mode, subscript, new_shifts)
                        )
                    case _RealSubscriptsInfo(s):
                        return self._compile_slice_chain(expr, subscripts)
            case Plus(children):
                ret_type = self._type_cache.get_type(children[0])
                ret = self.compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_value = self.compile_expr(child, subscripts)
                    child_type = self._type_cache.get_type(child)
                    ret = self._add(ret, ret_type, child_value, child_type, expr_type)
                    # the accumulator now has the result type; use it for the next
                    # operand instead of the stale type of the first child
                    ret_type = expr_type
                return ret
            case Times(children):
                ret_type = self._type_cache.get_type(children[0])
                ret = self.compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_value = self.compile_expr(child, subscripts)
                    child_type = self._type_cache.get_type(child)
                    ret = self._mul(ret, ret_type, child_value, child_type, expr_type)
                    ret_type = expr_type
                return ret
            case Power(_, exponent):
                base_type = self._type_cache.get_type(expr.base)
                base = self.compile_expr(expr.base, subscripts)
                match exponent:
                    case Int(exp_value):
                        return self._int_pow(base, base_type, exp_value, expr_type)
                    case Rational(num, 2):
                        s = self._sqrt(h.coerce(self._block, base, base_type, expr_type), expr_type)
                        return self._int_pow(s, expr_type, num, expr_type)
                    case _:
                        base = h.coerce(self._block, base, base_type, expr_type)
                        exp_type = self._type_cache.get_type(exponent)
                        exp = h.coerce(self._block, self.compile_expr(exponent, subscripts), exp_type, expr_type)
                        return self._pow(base, base_type, exp, exp_type, expr_type)
            case Sin(expr):
                arg = self.compile_expr(expr, subscripts)
                arg = h.coerce(self._block, arg, self._type_cache.get_type(expr), expr_type)
                assert isinstance(expr_type, ap.FloatType), "sin currently only supports real types"
                assert not isinstance(arg, ComplexValue)
                return self._block.sin(arg)
            case Cos(expr):
                arg = self.compile_expr(expr, subscripts)
                arg = h.coerce(self._block, arg, self._type_cache.get_type(expr), expr_type)
                assert isinstance(expr_type, ap.FloatType), "sin currently only supports real types"
                assert not isinstance(arg, ComplexValue)
                return self._block.cos(arg)
            case Ln(expr):
                arg = self.compile_expr(expr, subscripts)
                arg = h.coerce(self._block, arg, self._type_cache.get_type(expr), expr_type)
                assert isinstance(expr_type, ap.FloatType), "sin currently only supports real types"
                assert not isinstance(arg, ComplexValue)
                return self._block.ln(arg)
            case Exp(expr):
                arg = self.compile_expr(expr, subscripts)
                arg = h.coerce(self._block, arg, self._type_cache.get_type(expr), expr_type)
                assert isinstance(expr_type, ap.FloatType), "sin currently only supports real types"
                assert not isinstance(arg, ComplexValue)
                return self._block.exp(arg)

        raise TypeError(f'unsupported expression: {expr}')

    def _compile_lvalue(self, expr: Expr, subscripts: _SubscriptsInfo) -> tuple[Value, LowerType]:
        match expr:
            case Symbol():
                sym = self._symbol_scope.get_symbol(expr)
                lower_type = self._type_cache.get_symbol_type(expr)
                assert sym is not None
                match sym:
                    case ScalarArgInfo(is_ref=True):
                        # a scalar reference can be assigned to: the pointer is the
                        # lvalue itself
                        return self._args[sym.value], lower_type
                    case ScalarArgInfo():
                        raise TypeError(f"cannot use {expr} as left-value")
                    case ArrayArgInfo():
                        match subscripts:
                            case _RealSubscriptsInfo(s):
                                return self._compile_array_symbol_access(sym, s), lower_type
                            case _StandardLayoutSubscriptInfo():
                                return self._block.get_element_ptr(self._args[sym.ptr], self._flat_array_index(expr, subscripts)), lower_type
            case Slice():
                match subscripts:
                    case _StandardLayoutSubscriptInfo(mode, subscript, shifts):
                        # record the fixed axes and recurse; the offset is applied
                        # at the array access using the base array's strides
                        new_shifts = dict(shifts)
                        rank = len(self._type_cache.get_shape(expr.expr))
                        for axis, index in expr.axes:
                            if axis < 0 or axis >= rank:
                                raise IndexError(
                                    f"slice axis {axis} is out of bounds for {expr.expr}"
                                )
                            new_shifts[axis] = index
                        return self._compile_lvalue(
                            expr.expr, _StandardLayoutSubscriptInfo(mode, subscript, new_shifts)
                        )
                    case _RealSubscriptsInfo(s):
                        return self._compile_slice_lvalue(expr, subscripts)
        raise ValueError(f"cannot use {expr} as left-value")

    def _expr_cache_key(self, expr: Expr, subscripts: _SubscriptsInfo) -> tuple[Any, ...]:
        match subscripts:
            case _RealSubscriptsInfo(s):
                return (expr, s)
            case _StandardLayoutSubscriptInfo(subscript=subscript, shifts=shifts):
                return (expr, subscript, tuple(sorted(shifts.items())))
            case _:
                raise TypeError(f"unexpected subscripts info {type(subscripts).__name__}")

    def compile_expr(self, expr: Expr, subscripts: _SubscriptsInfo) -> MaybeComplexValue:
        cache_key = self._expr_cache_key(expr, subscripts)
        if cache_key in self._expr_cache:
            return self._expr_cache[cache_key]
        result = self._compile_expr_no_cache(expr, subscripts)
        self._expr_cache[cache_key] = result
        return result

    def compile_non_complex_expr(self, expr: Expr, subscripts: _SubscriptsInfo) -> Value:
        ret = self.compile_expr(expr, subscripts)
        assert not isinstance(ret, ComplexValue)
        return ret

    def _compile_assignment(self, typed_expr: TypedAssignExpr, tid: Value):
        expr = typed_expr.expr

        if self._standard_layout is StandardLayoutMode.NONE:
            if len(typed_expr.shape) == 0:
                # scalar assignment: no array subscripts to unpack
                subscripts: _SubscriptsInfo = _RealSubscriptsInfo(())
            else:
                shape: list[Value] = []
                for i in typed_expr.shape:
                    type = self._type_cache.get_type(i)
                    assert isinstance(type, ap.IntType), f"integer type expected for shape, got {type}"
                    value = self.compile_non_complex_expr(i, _RealSubscriptsInfo(()))
                    shape.append(value)
                subscripts = _RealSubscriptsInfo(self._compile_unpack_subscripts(tuple(shape), tid))
        else:
            # standard layout: use the flat loop variable directly as the array index
            subscripts = _StandardLayoutSubscriptInfo(self._standard_layout, tid, {})
        lhs_ptr, lhs_type = self._compile_lvalue(expr.lhs, subscripts)

        rhs_value = self.compile_expr(expr.rhs, subscripts)
        rhs_type = self._type_cache.get_type(expr.rhs)

        result_value = None
        final_type = rhs_type
        result_type = get_peer_types(lhs_type, rhs_type)

        def make_lhs():
            return self._block.load(lhs_ptr)
        match expr.op:
            case '':
                result_value = rhs_value
            case '+':
                result_value = self._add(make_lhs(), lhs_type, rhs_value, rhs_type, result_type)
                final_type = result_type
            case '-':
                result_value = self._sub(make_lhs(), lhs_type, rhs_value, rhs_type, result_type)
                final_type = result_type
            case '*':
                result_value = self._mul(make_lhs(), lhs_type, rhs_value, rhs_type, result_type)
                final_type = result_type
            case '/':
                result_value = self._div(make_lhs(), lhs_type, rhs_value, rhs_type, result_type)
                final_type = result_type
            case _:
                raise ValueError(f"unknown op {expr.op}")
        result_value = self._helper.coerce(self._block, result_value, final_type, lhs_type)
        self._store(lhs_ptr, result_value)
        # the store may change the value of an expression read while an earlier
        # assignment was compiled (e.g. a scalar reference that is written and
        # then read by a later assignment), so drop the cached expression values
        self._expr_cache.clear()

    def compile_assignments(self, exprs: list[TypedAssignExpr], tid: Value):
        assert not self._finished
        self._finished = True

        for expr in exprs:
            self._compile_assignment(expr, tid)

        return self._block

class CompiledWrapper:
    @override
    def __init__(self, parent: 'JitCompiler', symbols: _SymbolScope, inner: CompiledBackendFunction, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> None:
        self._parent = parent
        self._symbols = symbols
        self._inner = inner
        self.standard_layout = standard_layout
        self._invoke = _gen_call_invoke(symbols, parent, standard_layout)

    def call(self, *values: Any) -> Any:
        """Call the compiled kernel with the arguments in the order of the symbols
        passed to ``compile_assignments``/``compile_reduction``."""
        return self._invoke(self, values)

    @override
    def __str__(self) -> str:
        return f"CompiledWrapper(symbols={self._symbols}, inner={self._inner})"

    def print_all(self):
        return self._inner.print_all()

class TypedReductionExpr:
    def __init__(self, expr: Expr, ctx: TypeResolver) -> None:
        self.expr = expr
        # get_shape returns the shape in natural axis order (outermost first,
        # like numpy's ``.shape``); the loop unpacking consumes it as-is
        self.shape = ctx.get_shape(expr)

    def total_size(self):
        return Times.make(self.shape).normalize()

class _AssignmentsKernel(LoopKernel):
    @override
    def __init__(self, parent: 'JitCompiler', args: list[Symbol], exprs: list[AssignExpr], type_resolver: TypeResolver, by_ref_symbols: set[Symbol], reduction: Expr | None = None, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> None:
        type_cache = type_resolver
        self._parent = parent
        self._type_cache = type_cache
        self._exprs: list[TypedAssignExpr] = [TypedAssignExpr(a, type_cache) for a in exprs]
        self._reduction: TypedReductionExpr | None = None
        self.reduction_type: LowerType | None = None
        if reduction is not None:
            self._reduction = TypedReductionExpr(reduction, type_cache)
            reduction_type = type_cache.get_type(reduction)
            if isinstance(reduction_type, ap.IntType):
                # integer sums follow the C convention of being signed
                reduction_type = ap.IntType(reduction_type.bits, True)
            self.reduction_type = reduction_type
        self._total_size = _check_and_get_total_size(self._exprs, self._reduction, type_cache)
        self._helper = CompileHelper(parent)
        self.symbol_scope = _SymbolScope(type_cache)
        for symbol in args:
            self.symbol_scope.add_symbol(symbol, by_ref_symbols)
        self._standard_layout = self._check_standard_layout(standard_layout)

    @staticmethod
    def _contains_indexing(expr: Expr) -> bool:
        """Whether an expression contains a roll, flip or a nested slice.  Sliced
        views are not standard layout, so they cannot appear below a slice in a
        standard layout kernel."""
        todo = [expr]
        while todo:
            elem = todo.pop()
            if isinstance(elem, (Roll, Slice, Flip)):
                return True
            todo.extend(elem.subexpressions())
        return False

    def _check_standard_layout(self, requested: StandardLayoutMode) -> StandardLayoutMode:
        """Validate that the expressions can be compiled with a linear (SIMD friendly)
        index: no rolls, no slices on interior axes, no broadcasting, and every array
        must be standard layout.  Falls back to :data:`StandardLayoutMode.NONE` when
        the expressions are not compatible."""
        if requested is StandardLayoutMode.NONE:
            return StandardLayoutMode.NONE
        if len(self._exprs) > 0:
            loop_rank = len(self._exprs[0].shape)
        else:
            assert self._reduction is not None
            loop_rank = len(self._reduction.shape)
        dim_of = self._type_cache.get_symbol_dimension
        failures: list[str] = []

        def check_slice_chain(expr: Slice) -> None:
            # the sliced sub-expression must itself be standard layout, i.e. free of
            # slices, rolls and flips, so that the axes of the slice equal the axes
            # of the underlying arrays
            if self._contains_indexing(expr.expr):
                failures.append(
                    f"nested slices are not supported in standard layout kernels: {expr}"
                )
                return
            expr_shape = self._type_cache.get_shape(expr.expr)
            rank = len(expr_shape)
            fixed = {axis for axis, _ in expr.axes}
            if rank != loop_rank + len(fixed):
                failures.append(
                    f"the sliced expression {expr.expr} has rank {rank} but the loop "
                    f"rank is {loop_rank}; broadcasting sliced expressions is not "
                    "supported in standard layout kernels"
                )
                return
            if requested is StandardLayoutMode.ROW_MAJOR:
                # the fixed axes must be a leading prefix so that the surviving
                # axes form the contiguous block iterated by the loop
                ok_axes = fixed == set(range(len(fixed)))
            else:
                ok_axes = fixed == set(range(rank - len(fixed), rank))
            if not ok_axes:
                failures.append(
                    f"slice axes {sorted(fixed)} are not compatible with the "
                    f"{requested.value} layout"
                )

        def walk(expr: Expr) -> None:
            if isinstance(expr, Roll):
                failures.append("np.roll is not supported in standard layout kernels")
            elif isinstance(expr, Flip):
                failures.append("np.flip is not supported in standard layout kernels")
            elif isinstance(expr, Coord):
                failures.append("coord is not supported in standard layout kernels")
            elif isinstance(expr, Slice):
                check_slice_chain(expr)
            elif isinstance(expr, SymbolShape):
                # shape values are scalars; they do not access array data
                return
            elif isinstance(expr, Symbol):
                dim = dim_of(expr)
                if dim != 0 and dim != loop_rank:
                    failures.append(
                        f"array {expr} has rank {dim} but the loop rank is {loop_rank}; "
                        "broadcasting arrays are not supported in standard layout kernels"
                    )
            else:
                for child in expr.subexpressions():
                    walk(child)

        for typed in self._exprs:
            walk(typed.expr.lhs)
            walk(typed.expr.rhs)
        if self._reduction is not None:
            walk(self._reduction.expr)
        if len(failures) > 0:
            return StandardLayoutMode.NONE
        return requested

    @override
    def get_index_type(self) -> IntType:
        return IntType(self._parent.index_type.bits)

    @override
    def get_args(self) -> tuple[LowerType, ...]:
        return self.symbol_scope.get_args()

    @override
    def compile_total_size(self, begin: BasicBlock, args: tuple[Value, ...]) -> tuple[BasicBlock, Value]:
        cp = _FunctionCompiler(self._parent, self._helper, args, begin, self.symbol_scope,
                               standard_layout=self._standard_layout)
        value = cp.compile_non_complex_expr(self._total_size, _RealSubscriptsInfo(()))
        type = cp._type_cache.get_type(self._total_size)
        assert isinstance(type, ap.IntType), f"integer type expected for total size, got {type}"
        return begin, value

    @override
    def compile_body(self, begin: BasicBlock, args: tuple[Value, ...], loop_var: Value, debug: DebugInterface) -> tuple[BasicBlock, MaybeComplexValue]:
        cp = _FunctionCompiler(self._parent, self._helper, args, begin, self.symbol_scope, debug=debug,
                               standard_layout=self._standard_layout)
        cp.compile_assignments(self._exprs, loop_var)
        value: MaybeComplexValue = VoidValue()
        if self._reduction is not None:
            if self._standard_layout is StandardLayoutMode.NONE:
                shape: list[Value] = []
                for i in self._reduction.shape:
                    size_type = cp._type_cache.get_type(i)
                    assert isinstance(size_type, ap.IntType), f"integer type expected for shape, got {size_type}"
                    shape.append(cp.compile_non_complex_expr(i, _RealSubscriptsInfo(())))
                subscripts: _SubscriptsInfo = _RealSubscriptsInfo(
                    cp._compile_unpack_subscripts(tuple(shape), loop_var) if len(shape) > 0 else ()
                )
            else:
                subscripts = _StandardLayoutSubscriptInfo(self._standard_layout, loop_var, {})
            value = cp.compile_expr(self._reduction.expr, subscripts)
        return begin, value

class SumReductionKernel(ReductionKernel):
    @override
    def __init__(self, type: LowerType, helper: CompileHelper) -> None:
        self.type = type
        self._helper = helper

    @override
    def get_type(self) -> LowerType:
        return self.type

    @override
    def store_initial_value(self, block: BasicBlock, value_ptr: Value):
        match self.type:
            case ComplexFloatType(type):
                llvm_float_type = type.to_llvm_type()
                re = block.get_element_ptr(value_ptr, 0, 0)
                im = block.get_element_ptr(value_ptr, 0, 1)
                block.store(re, llvm_float_type.from_int(0))
                block.store(im, llvm_float_type.from_int(0))
            case _:
                block.store(value_ptr, self.type.to_llvm_type().from_int(0))

    @override
    def reduce(self, block: BasicBlock, acc_ptr: Value, value: MaybeComplexValue, ordering: Ordering | None = None):
        match self.type:
            case ComplexFloatType():
                re_acc = block.get_element_ptr(acc_ptr, 0, 0)
                im_acc = block.get_element_ptr(acc_ptr, 0, 1)
                re_value, im_value = self._helper.expand_complex_value(block, value)
                if ordering is not None:
                    block.atomicrmw(FAdd(), re_acc, re_value, ordering)
                    block.atomicrmw(FAdd(), im_acc, im_value, ordering)
                else:
                    block.store(re_acc, block.add(block.load(re_acc), re_value))
                    block.store(im_acc, block.add(block.load(im_acc), im_value))
            case _:
                assert not isinstance(value, ComplexValue)
                if ordering is not None:
                    if isinstance(value.get_type(), FloatType):
                        block.atomicrmw(FAdd(), acc_ptr, value, ordering)
                    else:
                        block.atomicrmw(Add(), acc_ptr, value, ordering)
                else:
                    block.store(acc_ptr, block.add(block.load(acc_ptr), value))
        return block

_F64 = ap.FloatType(64)
_U64 = ap.IntType(64, False)

class JitCompiler(TypesConfig):
    def __init__(self, backend: Backend, real_type: ap.FloatType = _F64, index_type: ap.IntType = _U64):
        self._backend = backend
        self.real_type = real_type
        self.index_type = index_type

    def compile_assignments(self, args: list[Symbol], exprs: list[AssignExpr], type_resolver: TypeResolver, by_ref_symbols: set[Symbol], reduction: Expr | None = None, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> CompiledWrapper:
        # scalar references that are never written to do not need a pointer: pass
        # them by value so reads do not pay a pointer indirection
        written = _collect_lvalue_symbols(exprs)
        effective_by_ref = {
            sym for sym in by_ref_symbols
            if sym in written and type_resolver.symbol_types[sym].dimension == 0
        }
        kernel = _AssignmentsKernel(self, args, exprs, type_resolver, effective_by_ref, reduction, standard_layout)
        reduction_kernel: ReductionKernel | None = None
        if reduction is not None:
            assert kernel.reduction_type is not None
            reduction_kernel = SumReductionKernel(kernel.reduction_type, kernel._helper)
        compiled = self._backend.compile_paralell_loop(kernel, reduction_kernel)

        return CompiledWrapper(self, kernel.symbol_scope, compiled, standard_layout=kernel._standard_layout)

    def compile_reduction(self, args: list[Symbol], expr: Expr, type_resolver: TypeResolver, by_ref_symbols: set[Symbol] | None = None, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> CompiledWrapper:
        return self.compile_assignments(args, [], type_resolver, by_ref_symbols or set(), reduction=expr, standard_layout=standard_layout)
