from enum import Enum
from typing import Any, override

import ctypes

from . import argpass as ap

from .helper import CompileHelper, ComplexValue, MaybeComplexValue
from ..expr import AssignExpr, Cos, Exp, Expr, Float, Int, Ln, Plus, Power, Rational, Roll, Sin, Slice, Symbol, Times
from .argpass import ArrayArgInfo, ComplexFloatType, ScalarArgInfo, SymbolArgInfo, TypesConfig, LowerType, TypeCache, TypeContext, TypedAssignExpr, get_peer_type
from .backend import Backend, CompiledBackendFunction, DebugInterface, LoopKernel
from .llvm import BasicBlock, FloatValue, IntType, IntValue, Value

def _check_and_get_total_size(exprs: list[TypedAssignExpr]):
    first_size = exprs[0].total_size()
    for expr in exprs[1:]:
        expr_size = expr.total_size()
        assert first_size == expr_size, f"incompatible expressions {exprs[0]} and {expr}, with incompatible total sizes {first_size} and {expr_size}"

    assert first_size is not None, "cannot compile expression with unspecified shapes"

    return first_size

class StandardLayoutMode(Enum):
    NONE = "none"
    COLUMN_MAJOR = "column"
    ROW_MAJOR = "row"

class _SymbolScope:
    _parent: 'JitCompiler'
    type_cache: TypeCache
    _symbol_values: dict[Symbol, SymbolArgInfo]
    _args: list[LowerType]
    _helper: CompileHelper

    def __init__(self, parent: 'JitCompiler', helper: CompileHelper, type_cache: TypeCache) -> None:
        self._parent = parent
        self._symbol_values = {}
        self._args = []
        self._helper = helper
        self.type_cache = type_cache

    def get_symbol(self, symbol: Symbol):
        return self._symbol_values[symbol]

    def add_symbol(self, symbol: Symbol, info: SymbolArgInfo):
        assert symbol not in self._symbol_values
        self._symbol_values[symbol] = info

    def get_args(self) -> tuple[LowerType, ...]:
        return tuple(self._args)

    def get_arg_count(self) -> int:
        return len(self._args)

    def items(self):
        return self._symbol_values.items()

    def _add_arg(self, type: LowerType):
        ret = len(self._args)
        self._args.append(type)
        return ret

    def _add_symbol(self, expr: Symbol, is_ref: bool):
        if expr in self._symbol_values:
            return
        lower_type = self.type_cache.get_symbol_type(expr)
        shape = self.type_cache.get_symbol_shape(expr)
        assert shape is not None, f"cannot compile symbol {expr} with unspecified shape"
        if len(shape) == 0:
            if is_ref:
                ret = ScalarArgInfo(
                    self._add_arg(ap.PointerType(lower_type)),
                    True,
                )
                self._symbol_values[expr] = ret
            else:
                ret = ScalarArgInfo(
                    self._add_arg(lower_type),
                    False,
                )
                self._symbol_values[expr] = ret
        else:
            # TODO: check indices types
            ret = ArrayArgInfo(
                self._add_arg(ap.PointerType(lower_type)),
                tuple(self._add_arg(self._parent.index_type) for _ in range(len(shape))),
            )
            self._symbol_values[expr] = ret

    def scan_lvalue_symbols(self, expr: Expr):
        match expr:
            case Symbol():
                self._add_symbol(expr, True)
            case Roll():
                self.scan_lvalue_symbols(expr.expr)
            case Slice():
                self.scan_lvalue_symbols(expr.expr)

    def scan_symbols(self, expr: Expr):
        todo = [expr]
        while len(todo) > 0:
            elem = todo.pop()
            if isinstance(elem, Symbol):
                self._add_symbol(elem, False)
            else:
                children = elem.subexpressions()
                children.reverse()
                todo.extend(children)

    def scan_assignment(self, typed_expr: TypedAssignExpr):
        expr = typed_expr.expr
        self.scan_lvalue_symbols(expr.lhs)
        self.scan_symbols(expr.lhs)
        self.scan_symbols(expr.rhs)

        for e in typed_expr.shape:
            self.scan_symbols(e)

    def scan_assignments(self, exprs: list[TypedAssignExpr]):
        for expr in exprs:
            self.scan_assignment(expr)

    @override
    def __str__(self):
        elems: list[str] = []
        for sym, info in self._symbol_values.items():
            match info:
                case ScalarArgInfo():
                    elems.append(f"%{info.value}: {'&' if info.is_ref else ''}Scalar = {sym}")
                case ArrayArgInfo():
                    strides = ', '.join(f"%{s}" for s in info.strides)
                    elems.append(f"%{info.ptr}: Array(strides=({strides})) = {sym}")
        return f"SymbolScope({', '.join(elems)})"

class _FunctionCompiler:
    parent: 'JitCompiler'
    _helper: CompileHelper
    _expr_cache: dict[tuple[Expr, tuple[Value, ...]], tuple[MaybeComplexValue, LowerType]]
    _subscript_cache: dict[tuple[tuple[Value, ...], tuple[Value, ...]], Value]
    _block: BasicBlock
    _finished: bool
    _layout_mode: StandardLayoutMode
    _symbol_scope: _SymbolScope
    _args: tuple[Value, ...]
    _type_cache: TypeCache
    _debug: DebugInterface | None

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
        self._expr_cache = {}
        self._subscript_cache = {}
        self._finished = False
        self._type_cache = symbol_scope.type_cache
        self._standard_layout = standard_layout
        self._debug = debug

    def _add(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType) -> tuple[MaybeComplexValue, ap.LowerType]:
        result_type = get_peer_type(left_type, right_type)
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_add(self._block, left, right), result_type
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.add(left, right), result_type

    def _sub(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType) -> tuple[MaybeComplexValue, ap.LowerType]:
        result_type = get_peer_type(left_type, right_type)
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_sub(self._block, left, right), result_type
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.sub(left, right), result_type

    def _mul(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType) -> tuple[MaybeComplexValue, ap.LowerType]:
        result_type = get_peer_type(left_type, right_type)
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_mul(self._block, left, right), result_type
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.mul(left, right), result_type

    def _div(self, left: MaybeComplexValue, left_type: ap.LowerType, right: MaybeComplexValue, right_type: ap.LowerType) -> tuple[MaybeComplexValue, ap.LowerType]:
        result_type = get_peer_type(left_type, right_type)
        left = self._helper.coerce(self._block, left, left_type, result_type)
        right = self._helper.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            return self._helper.complex_div(self._block, left, right), result_type
        assert not isinstance(left, ComplexValue) and not isinstance(right, ComplexValue)
        return self._block.div(left, right, True), result_type

    def _sqrt(self, expr: MaybeComplexValue, type: ap.LowerType) -> tuple[MaybeComplexValue, ap.LowerType]:
        if isinstance(type, ComplexFloatType):
            raise NotImplementedError
        assert not isinstance(expr, ComplexValue)
        return self._block.sqrt(expr), type

    def _pow(self, base: MaybeComplexValue, base_type: ap.LowerType, exp: MaybeComplexValue, exp_type: ap.LowerType) -> tuple[MaybeComplexValue, ap.LowerType]:
        result_type = get_peer_type(base_type, exp_type)
        base = self._helper.coerce(self._block, base, base_type, result_type)
        exp = self._helper.coerce(self._block, exp, result_type, result_type)
        if isinstance(result_type, ComplexFloatType):
            raise NotImplementedError
        assert not isinstance(base, ComplexValue) and not isinstance(exp, ComplexValue)
        return self._block.pow(base, exp), result_type

    def _store(self, ptr: Value, value: MaybeComplexValue):
        b = self._block
        match value:
            case ComplexValue(re, im):
                b.store(b.get_element_ptr(ptr, 0, 0), re)
                b.store(b.get_element_ptr(ptr, 0, 1), im)
            case _:
                b.store(ptr, value)

    def _compile_unpack_subscripts(self, sizes: tuple[Value, ...], packed: Value) -> tuple[Value, ...]:
        assert len(sizes) > 0
        ret: list[Value] = []
        for size in sizes[-1:0:-1]:
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

    def _promote_type(self, type: LowerType) -> LowerType:
        match type:
            case ap.IntType():
                return self.parent.index_type
            case ap.FloatType():
                return self.parent.real_type
            case ap.ComplexFloatType(type):
                t = self._promote_type(type)
                assert isinstance(t, ap.FloatType)
                return ap.ComplexFloatType(t)
            case _:
                raise ValueError(f"unsupported type: {type}")

    def _compile_expr_no_cache(self, expr: Expr, subscripts: tuple[Value, ...]) -> tuple[MaybeComplexValue, LowerType]:
        h = self._helper

        match expr:
            case Int(value):
                return IntValue(value, h.llvm_index_type), self.parent.index_type
            case Rational(numerator, denominator):
                return FloatValue(numerator / denominator, h.llvm_real_type), self.parent.real_type
            case Float(value):
                return FloatValue(value, h.llvm_real_type), self.parent.real_type
            case Symbol():
                sym = self._symbol_scope.get_symbol(expr)
                lower_type = self._type_cache.get_symbol_type(expr)
                lower_target_type = self._promote_type(lower_type)
                ret = None
                match sym:
                    case ScalarArgInfo():
                        ret = self._block.load(self._args[sym.value]) if sym.is_ref else self._args[sym.value]
                    case ArrayArgInfo():
                        ret = self._block.load(self._compile_array_symbol_access(sym, subscripts))
                    case _:
                        raise NotImplementedError
                return self._helper.coerce(self._block, ret, lower_type, lower_target_type), lower_target_type
            case Roll():
                assert not self._standard_layout, "cannot compile Roll in standard layout mode"
                expr_shape = self._type_cache.get_shape(expr.expr)
                assert expr_shape is not None, "cannot compile unspecified shape"
                len, len_type = self.compile_non_complex_expr(expr_shape[expr.axis], ())
                assert isinstance(len_type, ap.IntType), "length must be an integer"
                new_index = subscripts[expr.axis]
                new_index = self._block.add(new_index, IntValue(-expr.amount, h.llvm_index_type))
                new_index = self._block.rem(new_index, len, False)
                subscripts = subscripts[:expr.axis] + (new_index,) + subscripts[expr.axis + 1:]
                return self.compile_expr(expr.expr, subscripts)
            case Slice():
                return self.compile_expr(expr.expr, subscripts[:expr.axis] + (IntValue(expr.index, h.llvm_index_type),) + subscripts[expr.axis + 1:])
            case Plus(children):
                ret, ret_type = self.compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_value, child_type = self.compile_expr(child, subscripts)
                    ret, ret_type = self._add(ret, ret_type, child_value, child_type)
                return ret, ret_type
            case Times(children):
                ret, ret_type = self.compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_value, child_type = self.compile_expr(child, subscripts)
                    ret, ret_type = self._mul(ret, ret_type, child_value, child_type)
                return ret, ret_type
            case Power(_, exponent):
                base, base_type = self.compile_expr(expr.base, subscripts)
                match exponent:
                    case Int(exp_value):
                        neg = False
                        if exp_value < 0:
                            exp_value = -exp_value
                            neg = True
                            base, base_type = h.coerce_int_to_float(self._block, base, base_type, self.parent.real_type)
                        ret = base
                        for _ in range(exp_value - 1):
                            ret, _ = self._mul(ret, base_type, base, base_type)
                        if neg:
                            ret, _ = self._div(base_type.to_llvm_type().from_int(1), base_type, ret, base_type)
                        return ret, base_type
                    case Rational(1, 2):
                        base, base_type = h.coerce_int_to_float(self._block, base, base_type, self.parent.real_type)
                        return self._sqrt(base, base_type)
                    case Rational(-1, 2):
                        base, base_type = h.coerce_int_to_float(self._block, base, base_type, self.parent.real_type)
                        base2, base_type = self._sqrt(base, base_type)
                        return self._div(base_type.to_llvm_type().from_int(1), base_type, base2, base_type)
                    case _:
                        exp_value, exp_type = self.compile_expr(exponent, subscripts)
                        return self._pow(base, base_type, exp_value, exp_type)
            case Sin(expr):
                arg, arg_type = self.compile_expr(expr, subscripts)
                arg, arg_type = h.coerce_int_to_float(self._block, arg, arg_type, self.parent.real_type)
                assert not isinstance(arg_type, ComplexFloatType), "sin currently only supports real types"
                assert not isinstance(arg, MaybeComplexValue), "sin currently only supports real types"
                return self._block.sin(arg), arg_type
            case Cos(expr):
                arg, arg_type = self.compile_expr(expr, subscripts)
                arg, arg_type = h.coerce_int_to_float(self._block, arg, arg_type, self.parent.real_type)
                assert not isinstance(arg_type, ComplexFloatType), "cos currently only supports real types"
                assert not isinstance(arg, MaybeComplexValue), "cos currently only supports real types"
                return self._block.cos(arg), arg_type
            case Ln(expr):
                arg, arg_type = self.compile_expr(expr, subscripts)
                arg, arg_type = h.coerce_int_to_float(self._block, arg, arg_type, self.parent.real_type)
                assert not isinstance(arg_type, ComplexFloatType), "ln currently only supports real types"
                assert not isinstance(arg, MaybeComplexValue), "ln currently only supports real types"
                return self._block.ln(arg), arg_type
            case Exp(expr):
                arg, arg_type = self.compile_expr(expr, subscripts)
                arg, arg_type = h.coerce_int_to_float(self._block, arg, arg_type, self.parent.real_type)
                assert not isinstance(arg_type, ComplexFloatType), "exp currently only supports real types"
                assert not isinstance(arg, MaybeComplexValue), "exp currently only supports real types"
                return self._block.exp(arg), arg_type

        raise TypeError(f'unsupported expression: {expr}')

    def _compile_lvalue(self, expr: Expr, subscripts: tuple[Value, ...]) -> tuple[Value, LowerType]:
        match expr:
            case Symbol():
                sym = self._symbol_scope.get_symbol(expr)
                lower_type = self._type_cache.get_symbol_type(expr)
                assert sym is not None
                match sym:
                    case ScalarArgInfo():
                        if sym.is_ref:
                            return self._args[sym.value], lower_type
                        else:
                            raise TypeError(f"cannot use {expr} as left-value")
                    case ArrayArgInfo():
                        return self._compile_array_symbol_access(sym, subscripts), lower_type
        raise ValueError(f"cannot use {expr} as left-value")

    def compile_expr(self, expr: Expr, subscripts: tuple[Value, ...]) -> tuple[MaybeComplexValue, LowerType]:
        cache_key = (expr, subscripts)
        if cache_key in self._expr_cache:
            return self._expr_cache[cache_key]
        result = self._compile_expr_no_cache(expr, subscripts)
        self._expr_cache[cache_key] = result
        return result

    def compile_non_complex_expr(self, expr: Expr, subscripts: tuple[Value, ...]) -> tuple[Value, LowerType]:
        ret, type = self.compile_expr(expr, subscripts)
        assert not isinstance(ret, ComplexValue)
        return ret, type

    def _compile_assignment(self, typed_expr: TypedAssignExpr, tid: Value):
        expr = typed_expr.expr

        shape: list[Value] = []
        for i in typed_expr.shape:
            value, type = self.compile_non_complex_expr(i, ())
            assert isinstance(type, ap.IntType), f"integer type expected for shape, got {type}"
            shape.append(value)
        indices = self._compile_unpack_subscripts(tuple(shape), tid)
        lhs_ptr, lhs_type = self._compile_lvalue(expr.lhs, indices)

        rhs_value, rhs_type = self.compile_expr(expr.rhs, indices)

        result_value = None
        match expr.op:
            case '':
                result_value = rhs_value, rhs_type
            case '+':
                result_value = self._add(self._block.load(lhs_ptr), lhs_type, rhs_value, rhs_type)
            case '-':
                result_value = self._sub(self._block.load(lhs_ptr), lhs_type, rhs_value, rhs_type)
            case '*':
                result_value = self._mul(self._block.load(lhs_ptr), lhs_type, rhs_value, rhs_type)
            case '/':
                result_value = self._div(self._block.load(lhs_ptr), lhs_type, rhs_value, rhs_type)
            case _:
                raise ValueError(f"unknown op {expr.op}")
        result_value = self._helper.coerce_lower_type(self._block, result_value[0], result_value[1], lhs_type)
        self._store(lhs_ptr, result_value)

    def compile_assignments(self, exprs: list[TypedAssignExpr], tid: Value):
        assert not self._finished
        self._finished = True

        for expr in exprs:
            self._compile_assignment(expr, tid)

        return self._block

class CompiledWrapper:
    _parent: 'JitCompiler'
    _symbols: _SymbolScope
    _inner: CompiledBackendFunction

    @override
    def __init__(self, parent: 'JitCompiler', symbols: _SymbolScope, inner: CompiledBackendFunction) -> None:
        self._parent = parent
        self._symbols = symbols
        self._inner = inner

    def call(self, arg: dict[Symbol, Any]) -> None:
        index_type = self._parent.index_type.to_ctype()
        seen_symbols: set[Symbol] = set()
        converted_args: list[ctypes._CDataType | None] = [None for _ in range(self._symbols.get_arg_count())]
        for symbol, value in arg.items():
            seen_symbols.add(symbol)
            info = self._symbols.get_symbol(symbol)
            lower_type = self._symbols.type_cache.get_symbol_type(symbol)
            lower_type_ctype = lower_type.to_ctype()
            match info:
                case ScalarArgInfo():
                    if info.is_ref:
                        raise NotImplementedError
                    else:
                        converted_args[info.value] = lower_type_ctype(value)
                case ArrayArgInfo():
                    ptr_type = ctypes.POINTER(lower_type_ctype)
                    # np.ndarray
                    value_strides = value.strides
                    converted_args[info.ptr] = ctypes.cast(value.ctypes.data, ptr_type)
                    assert len(value_strides) == len(info.strides)
                    for index, stride in zip(info.strides, value_strides):
                        converted_args[index] = index_type(stride // ctypes.sizeof(lower_type_ctype))

        for symbol in self._symbols._symbol_values:
            if symbol not in seen_symbols:
                raise ValueError(f"Symbol {symbol} not found in arg")
        for a in converted_args:
            assert a is not None
        self._inner.call(*converted_args) # type: ignore

    @override
    def __str__(self) -> str:
        return f"CompiledWrapper(symbols={self._symbols}, inner={self._inner})"

    def print_all(self):
        return self._inner.print_all()

class _AssignmentsKernel(LoopKernel):
    _parent: 'JitCompiler'
    _exprs: list[TypedAssignExpr]
    _symbol_scope: _SymbolScope
    _total_size: Expr
    _helper: CompileHelper

    @override
    def __init__(self, parent: 'JitCompiler', exprs: list[AssignExpr], type_context: TypeContext) -> None:
        type_cache = TypeCache(type_context)
        self._parent = parent
        self._exprs = list(TypedAssignExpr(a, type_cache) for a in exprs)
        self._total_size = _check_and_get_total_size(self._exprs)
        self._helper = CompileHelper(parent)
        self._symbol_scope = _SymbolScope(parent, self._helper, type_cache)
        self._symbol_scope.scan_assignments(self._exprs)

    @override
    def get_index_type(self) -> IntType:
        return IntType(self._parent.index_type.bits)

    @override
    def get_args(self) -> tuple[LowerType, ...]:
        return self._symbol_scope.get_args()

    @override
    def compile_total_size(self, begin: BasicBlock, args: tuple[Value, ...]) -> tuple[BasicBlock, Value]:
        cp = _FunctionCompiler(self._parent, self._helper, args, begin, self._symbol_scope)
        value, type = cp.compile_non_complex_expr(self._total_size, ())
        assert isinstance(type, ap.IntType), f"integer type expected for total size, got {type}"
        return begin, value

    @override
    def compile_body(self, begin: BasicBlock, args: tuple[Value, ...], loop_var: Value, debug: DebugInterface) -> BasicBlock:
        cp = _FunctionCompiler(self._parent, self._helper, args, begin, self._symbol_scope, debug=debug)
        cp.compile_assignments(self._exprs, loop_var)
        return begin

class JitCompiler(TypesConfig):
    _backend: Backend

    def __init__(self, backend: Backend, real_type: ap.FloatType = ap.FloatType(64), index_type: ap.IntType = ap.IntType(64, False)):
        self._backend = backend
        self.real_type = real_type
        self.index_type = index_type

    def compile_one_kernel(self, exprs: list[AssignExpr], type_context: TypeContext) -> CompiledWrapper:
        kernel = _AssignmentsKernel(self, exprs, type_context)
        compiled = self._backend.compile_paralell_loop(kernel)

        return CompiledWrapper(self, kernel._symbol_scope, compiled)
