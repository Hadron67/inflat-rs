import ctypes
from dataclasses import dataclass
from enum import Enum
from typing import Any, override

import numpy as np

from ..expr import (
    AssignExpr,
    Complex,
    Cos,
    Exp,
    Expr,
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
    ArrayArgInfo,
    ComplexFloatType,
    LowerType,
    ScalarArgInfo,
    SymbolArgInfo,
    SymbolShape,
    TypeContext,
    TypedAssignExpr,
    TypeResolver,
    TypesConfig,
    get_peer_types,
)


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

class StandardLayoutMode(Enum):
    NONE = "none"
    COLUMN_MAJOR = "column"
    ROW_MAJOR = "row"

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

    def add_symbol(self, symbol: Symbol):
        """Register one function argument.  The registration order is the positional
        argument order of the compiled function."""
        if symbol in self._symbol_values:
            raise ValueError(f"duplicate symbol {symbol} in function arguments")
        lower_type = self.type_cache.get_symbol_type(symbol)
        dim = self.type_cache.get_symbol_dimension(symbol)
        if dim == 0:
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

    def _flat_array_index(self, sym: ArrayArgInfo, subscripts: _StandardLayoutSubscriptInfo) -> Value:
        """Compute the linear index of an array in standard layout mode: the flat
        subscript plus, for every sliced axis, the (normalized) slice index times the
        stride of that axis."""
        index = subscripts.subscript
        for axis, raw_index in subscripts.shifts.items():
            length = self._args[sym.shape[axis]]
            idx = self._normalize_slice_index(length, raw_index)
            stride = self._args[sym.strides[axis]]
            index = self._block.add(index, self._block.mul(idx, stride))
        return index

    def _compile_slice_chain(self, expr: Slice, subscripts: _RealSubscriptsInfo) -> MaybeComplexValue:
        """Compile a slice chain by building the subscript vector of the sliced
        expression directly and compiling the sliced expression with it, so compound
        bases like ``np.sin(b)[2]`` work.  Unlike the positional substitution, this
        also works when the sliced expression has a different rank than the loop,
        e.g. ``a += b[1]`` with a one-dimensional ``a``."""
        nodes: list[Slice] = []
        cur = expr
        while isinstance(cur, Slice):
            nodes.append(cur)
            cur = cur.expr
        cur_shape = self._type_cache.get_shape(cur)
        dim = len(cur_shape)
        # the axis attribute of a slice is relative to the expression it wraps, so
        # map each node to the axis of the sliced expression it fixes, innermost
        # first
        remaining = list(range(dim))
        node_axes: list[int] = []
        for node in reversed(nodes):
            if node.axis < 0 or node.axis >= len(remaining):
                raise TypeError(f"slice axis {node.axis} is out of bounds for {cur}")
            node_axes.append(remaining[node.axis])
            del remaining[node.axis]
        node_axes.reverse()
        r = len(subscripts.subscripts)
        entries: list[tuple[int, Value]] = []
        for node, axis in zip(nodes, node_axes):
            length = self.compile_non_complex_expr(cur_shape[axis], _RealSubscriptsInfo(()))
            entries.append((axis, self._normalize_slice_index(length, node.index)))
        if dim == r:
            # the loop covers every axis of the sliced expression: replace the value
            # at the fixed axes
            v: list[Value] = list(subscripts.subscripts)
            for axis, index in entries:
                v[axis] = index
            return self.compile_expr(cur, _RealSubscriptsInfo(tuple(v)))
        # the loop does not cover every axis: the sliced expression is broadcast, so
        # the surviving axes (ascending) receive the trailing-aligned loop subscripts
        # and the fixed axes receive their indices
        fixed = {axis for axis, _ in entries}
        surviving = [axis for axis in range(dim) if axis not in fixed]
        r_rhs = len(surviving)
        value_at = {axis: index for axis, index in entries}
        for j, axis in enumerate(surviving):
            value_at[axis] = subscripts.subscripts[r - r_rhs + j]
        v = [value_at[axis] for axis in range(dim)]
        return self.compile_expr(cur, _RealSubscriptsInfo(tuple(v)))

    def _compile_unpack_subscripts(self, sizes: tuple[Value, ...], packed: Value) -> tuple[Value, ...]:
        """
            sizes are innermost-first, i.e. in the order produced by merge_shape
        """
        assert len(sizes) > 0
        ret: list[Value] = []
        for size in sizes[:-1]:
            ret.append(self._block.rem(packed, size, False))
            packed = self._block.div(packed, size, False)
        ret.append(packed)
        return tuple(ret[-1::-1])

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
                    case ScalarArgInfo():
                        ret = self._args[sym.value]
                    case ArrayArgInfo():
                        match subscripts:
                            case _RealSubscriptsInfo(s):
                                ret = self._block.load(self._compile_array_symbol_access(sym, s))
                            case _StandardLayoutSubscriptInfo():
                                # standard layout: the loop variable is the linear array index
                                ret = self._block.load(
                                    self._block.get_element_ptr(self._args[sym.ptr], self._flat_array_index(sym, subscripts))
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
            case Roll():
                assert isinstance(subscripts, _RealSubscriptsInfo), "cannot compile Roll in standard layout mode"
                expr_shape = self._type_cache.get_shape(expr.expr)
                assert expr_shape is not None, "cannot compile unspecified shape"
                length = self.compile_non_complex_expr(expr_shape[expr.axis], _RealSubscriptsInfo(()))
                length_type = self._type_cache.get_type(expr_shape[expr.axis])
                assert isinstance(length_type, ap.IntType), "length must be an integer"
                # new index = (subscript - amount) mod length; add enough multiples
                # of the length so that the unsigned remainder is well-defined
                amount = expr.amount
                abs_amount = IntValue(abs(amount), h.llvm_index_type)
                # ceil(abs_amount / length)
                multiples = self._block.div(
                    self._block.add(abs_amount, self._block.sub(length, IntValue(1, h.llvm_index_type))),
                    length,
                    False,
                )
                new_index = self._block.add(
                    self._block.add(subscripts.subscripts[expr.axis], IntValue(-amount, h.llvm_index_type)),
                    self._block.mul(multiples, length),
                )
                new_index = self._block.rem(new_index, length, False)
                new_subscripts = _RealSubscriptsInfo(
                    subscripts.subscripts[:expr.axis] + (new_index,) + subscripts.subscripts[expr.axis + 1:]
                )
                return self.compile_expr(expr.expr, new_subscripts)
            case Slice():
                match subscripts:
                    case _StandardLayoutSubscriptInfo(mode, subscript, shifts):
                        # record the fixed axis and recurse; the offset is applied at
                        # the array access using the base array's strides
                        new_shifts = dict(shifts)
                        new_shifts[expr.axis] = expr.index
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
                return ret
            case Times(children):
                ret_type = self._type_cache.get_type(children[0])
                ret = self.compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_value = self.compile_expr(child, subscripts)
                    child_type = self._type_cache.get_type(child)
                    ret = self._mul(ret, ret_type, child_value, child_type, expr_type)
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
                    case ScalarArgInfo():
                        raise TypeError(f"cannot use {expr} as left-value")
                    case ArrayArgInfo():
                        match subscripts:
                            case _RealSubscriptsInfo(s):
                                return self._compile_array_symbol_access(sym, s), lower_type
                            case _StandardLayoutSubscriptInfo():
                                return self._block.get_element_ptr(self._args[sym.ptr], self._flat_array_index(sym, subscripts)), lower_type
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
            shape: list[Value] = []
            for i in typed_expr.shape:
                type = self._type_cache.get_type(i)
                assert isinstance(type, ap.IntType), f"integer type expected for shape, got {type}"
                value = self.compile_non_complex_expr(i, _RealSubscriptsInfo(()))
                shape.append(value)
            subscripts: _SubscriptsInfo = _RealSubscriptsInfo(self._compile_unpack_subscripts(tuple(shape), tid))
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

    def _check_layout(self, values) -> bool:
        """Verify that every array argument matches the layout the kernel was compiled for."""
        flag = 'C_CONTIGUOUS' if self.standard_layout is StandardLayoutMode.ROW_MAJOR else 'F_CONTIGUOUS'
        for value in values:
            if isinstance(value, np.ndarray) and not value.flags[flag]:
                return False
        return True

    def call(self, *values: Any) -> Any:
        """Call the compiled kernel with the arguments in the order of the symbols
        passed to ``compile_assignments``/``compile_reduction``."""
        arg_count = len(self._symbols.get_symbol_order())
        if len(values) != arg_count:
            raise TypeError(
                f"the kernel expects {arg_count} positional argument(s), got {len(values)}"
            )
        if self.standard_layout is not StandardLayoutMode.NONE and not self._check_layout(values):
            raise ValueError(
                f"the kernel was compiled for {self.standard_layout.value} layout but the array "
                "arguments do not match; pass contiguous arrays of the expected layout or recompile "
                "with standard_layout=StandardLayoutMode.NONE"
            )
        index_type = self._parent.index_type.to_ctype()
        converted_args: list[ctypes._CDataType | None] = [None for _ in range(self._symbols.get_arg_count())]
        for symbol, value in zip(self._symbols.get_symbol_order(), values):
            info = self._symbols.get_symbol(symbol)
            lower_type = self._symbols.type_cache.get_symbol_type(symbol)
            lower_type_ctype = lower_type.to_ctype()
            lower_type_size = ctypes.sizeof(lower_type_ctype)
            match info:
                case ScalarArgInfo():
                    converted_args[info.value] = lower_type_ctype(value)
                case ArrayArgInfo():
                    value_shape = value.shape
                    ptr_type = ctypes.POINTER(lower_type_ctype)
                    # np.ndarray
                    value_strides = value.strides
                    converted_args[info.ptr] = ctypes.cast(value.ctypes.data, ptr_type)
                    assert len(value_shape) == len(info.strides)
                    assert len(value_strides) == len(info.strides)
                    for index, shape in zip(info.shape, value_shape):
                        converted_args[index] = index_type(shape)
                    for index, stride in zip(info.strides, value_strides):
                        assert stride % lower_type_size == 0
                        converted_args[index] = index_type(stride // lower_type_size)
        for a in converted_args:
            assert a is not None
        return self._inner.call(*converted_args) # type: ignore

    @override
    def __str__(self) -> str:
        return f"CompiledWrapper(symbols={self._symbols}, inner={self._inner})"

    def print_all(self):
        return self._inner.print_all()

class TypedReductionExpr:
    def __init__(self, expr: Expr, ctx: TypeResolver) -> None:
        self.expr = expr
        shape = ctx.get_shape(expr)
        # get_shape returns shape elements in an order that depends on how many
        # shape merges the expression underwent; normalize to the innermost-first
        # convention used by _compile_unpack_subscripts
        indices = [e.index for e in shape if isinstance(e, SymbolShape)]
        if len(indices) > 0 and indices == sorted(indices):
            shape = tuple(reversed(shape))
        self.shape = shape

    def total_size(self):
        return Times.make(self.shape).normalize()

class _AssignmentsKernel(LoopKernel):
    @override
    def __init__(self, parent: 'JitCompiler', args: list[Symbol], exprs: list[AssignExpr], type_context: TypeContext, reduction: Expr | None = None, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> None:
        type_cache = TypeResolver(type_context, parent)
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
            self.symbol_scope.add_symbol(symbol)
        self._standard_layout = self._check_standard_layout(standard_layout)

    @staticmethod
    def _contains_indexing(expr: Expr) -> bool:
        """Whether an expression contains a roll or a nested slice.  Sliced views are
        not standard layout, so they cannot appear below a slice in a standard
        layout kernel."""
        todo = [expr]
        while todo:
            elem = todo.pop()
            if isinstance(elem, (Roll, Slice)):
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
            # slices and rolls, so that the axis attribute of the slice equals the
            # axis of the underlying arrays
            if self._contains_indexing(expr.expr):
                failures.append(
                    f"nested slices are not supported in standard layout kernels: {expr}"
                )
                return
            expr_shape = self._type_cache.get_shape(expr.expr)
            rank = len(expr_shape)
            if rank != loop_rank + 1:
                failures.append(
                    f"the sliced expression {expr.expr} has rank {rank} but the loop "
                    f"rank is {loop_rank}; broadcasting sliced expressions is not "
                    "supported in standard layout kernels"
                )
                return
            if requested is StandardLayoutMode.ROW_MAJOR:
                ok_axis = expr.axis == 0
            else:
                ok_axis = expr.axis == rank - 1
            if not ok_axis:
                failures.append(
                    f"slice axis {expr.axis} is not compatible with the "
                    f"{requested.value} layout"
                )

        def walk(expr: Expr) -> None:
            if isinstance(expr, Roll):
                failures.append("np.roll is not supported in standard layout kernels")
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

@dataclass(frozen=True)
class ArgType:
    type: LowerType
    rank: int
    is_ref: bool = False

class JitCompiler(TypesConfig):
    def __init__(self, backend: Backend, real_type: ap.FloatType = _F64, index_type: ap.IntType = _U64):
        self._backend = backend
        self.real_type = real_type
        self.index_type = index_type

    def compile_assignments(self, args: list[tuple[Symbol, ArgType]], exprs: list[AssignExpr], reduction: Expr | None = None, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> CompiledWrapper:
        type_context = TypeContext()
        for sym, type in args:
            type_context.set_symbol(sym, type.type, type.rank)
        kernel = _AssignmentsKernel(self, [a[0] for a in args], exprs, type_context, reduction, standard_layout)
        reduction_kernel: ReductionKernel | None = None
        if reduction is not None:
            assert kernel.reduction_type is not None
            reduction_kernel = SumReductionKernel(kernel.reduction_type, kernel._helper)
        compiled = self._backend.compile_paralell_loop(kernel, reduction_kernel)

        return CompiledWrapper(self, kernel.symbol_scope, compiled, standard_layout=kernel._standard_layout)

    def compile_reduction(self, args: list[tuple[Symbol, ArgType]], expr: Expr, standard_layout: StandardLayoutMode = StandardLayoutMode.NONE) -> CompiledWrapper:
        return self.compile_assignments(args, [], reduction=expr, standard_layout=standard_layout)
