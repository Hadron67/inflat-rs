from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import override, Callable

from .expr import AssignExpr, ComplexType, Cos, Exp, Expr, Float, Int, IntegerType, Ln, Plus, Power, Rational, RealType, Roll, Sin, Slice, Symbol, Times
from .llvm import AggregateValue, ArgValue, BasicBlock, Float64Type, FloatType, FloatValue, Function, IntType, IntValue, PointerType, StructType, Type, Value, VoidType

class StandardLayoutMode(Enum):
    NONE = "none"
    COLUMN_MAJOR = "column"
    ROW_MAJOR = "row"

class ArgInfo:
    @abstractmethod
    def map_arg(self, op: Callable[[int], int]) -> 'ArgInfo':
        raise NotImplementedError

@dataclass
class ScalarArgInfo(ArgInfo):
    value: int
    is_ref: bool

    @override
    def __str__(self) -> str:
        return f"%{self.value}: {'&' if self.is_ref else ''}Scalar"

    @override
    def map_arg(self, op: Callable[[int], int]) -> ArgInfo:
        return ScalarArgInfo(op(self.value), self.is_ref)

@dataclass
class ArrayArgInfo(ArgInfo):
    ptr: int
    strides: tuple[int, ...]

    def __str__(self) -> str:
        return f"%{self.ptr}: Array(strides=({", ".join(str(i) for i in self.strides)}))"

    @override
    def map_arg(self, op: Callable[[int], int]) -> 'ArgInfo':
        return ArrayArgInfo(op(self.ptr), tuple(op(i) for i in self.strides))

class FunctionCompiler:
    parent: 'JitCompiler'
    _expr_cache: dict[tuple[Expr, tuple[Value, ...]], Value]
    _subscript_cache: dict[tuple[tuple[Value, ...], tuple[Value, ...]], Value]
    _fn: Function
    _block: BasicBlock
    _symbol_values: dict[Symbol, ArgInfo]
    _tid: ArgValue | None
    _finished: bool
    _layout_mode: StandardLayoutMode
    _array_symbol_shapes: dict[Symbol, tuple[Value, ...]]

    def __init__(
        self,
        parent: 'JitCompiler',
        fn_name: str | None = None,
        standard_layout: StandardLayoutMode = StandardLayoutMode.NONE,
    ) -> None:
        self.parent = parent
        self._fn = Function(fn_name)
        self._block = self._fn.entry
        self._expr_cache = {}
        self._subscript_cache = {}
        self._symbol_values = {}
        self._array_symbol_shapes = {}
        self._tid = None
        self._finished = False
        self._standard_layout = standard_layout

    def _compile_add(self, left: Value, left_type: Expr, right: Value, right_type: Expr) -> Value:
        # TODO: complex
        result_type = combine_types(left_type, right_type)
        left = self.parent.coerce(self._block, left, left_type, result_type)
        right = self.parent.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexType):
            return self.parent.complex_add(self._block, left, right)
        return self._block.add(left, right)

    def _compile_mul(self, left: Value, left_type: Expr, right: Value, right_type: Expr) -> Value:
        # TODO: ditto
        result_type = combine_types(left_type, right_type)
        left = self.parent.coerce(self._block, left, left_type, result_type)
        right = self.parent.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexType):
            return self.parent.complex_mul(self._block, left, right)
        return self._block.mul(left, right)

    def _compile_div(self, left: Value, left_type: Expr, right: Value, right_type: Expr) -> Value:
        result_type = combine_types(left_type, right_type)
        left = self.parent.coerce(self._block, left, left_type, result_type)
        right = self.parent.coerce(self._block, right, right_type, result_type)
        if isinstance(result_type, ComplexType):
            return self.parent.complex_div(self._block, left, right)
        return self._block.div(left, right)

    def _compile_unpack_subscripts(self, sizes: tuple[Value, ...], packed: Value) -> tuple[Value, ...]:
        assert len(sizes) > 0
        ret: list[Value] = []
        for size in sizes[-1:0:-1]:
            ret.append(self._block.rem(packed, size))
            packed = self._block.div(packed, size)
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
        return self._block.get_array_element_ptr(
            self._fn.get_arg(info.ptr),
            self._compile_subscript(tuple(self._fn.get_arg(i) for i in info.strides), subscripts),
        )

    def _compile_expr_no_cache(self, expr: Expr, subscripts: tuple[Value, ...]) -> Value:
        jc = self.parent

        match expr:
            case Int(value):
                return IntValue(value, self.parent.index_type)
            case Rational(numerator, denominator):
                return FloatValue(numerator / denominator, self.parent.real_type)
            case Float(value):
                return FloatValue(value, self.parent.real_type)
            case Symbol():
                if expr not in self._symbol_values:
                    sym = self._add_symbol(expr, False)
                sym = self._symbol_values[expr]
                match sym:
                    case ScalarArgInfo():
                        if sym.is_ref:
                            return self._block.load(self._fn.get_arg(sym.value))
                        else:
                            return self._fn.get_arg(sym.value)
                    case ArrayArgInfo():
                        ret = self._block.load(self._compile_array_symbol_access(sym, subscripts))
                        return ret
            case Roll():
                assert not self._standard_layout, "cannot compile Roll in standard layout mode"
                expr_shape = expr.expr.get_shape()
                assert expr_shape is not None, "cannot compile unspecified shape"
                len = self._compile_expr(expr_shape[expr.axis], ())
                new_index = subscripts[expr.axis]
                new_index = self._block.add(new_index, IntValue(-expr.amount, jc.index_type))
                new_index = self._block.rem(new_index, len)
                subscripts = subscripts[:expr.axis] + (new_index,) + subscripts[expr.axis + 1:]
                return self._compile_expr(expr.expr, subscripts)
            case Slice():
                return self._compile_expr(expr.expr, subscripts[:expr.axis] + (IntValue(expr.index, jc.index_type),) + subscripts[expr.axis + 1:])
            case Plus(children):
                ret_type = children[0].get_type()
                ret = self._compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_type = child.get_type()
                    ret = self._compile_add(ret, ret_type, self._compile_expr(child, subscripts), child_type)
                    ret_type = child_type
                return ret
            case Times(children):
                ret_type = children[0].get_type()
                ret = self._compile_expr(children[0], subscripts)
                for child in children[1:]:
                    child_type = child.get_type()
                    ret = self._compile_mul(ret, ret_type, self._compile_expr(child, subscripts), child_type)
                    ret_type = child_type
                return ret
            case Power(_, exponent):
                base_type = expr.base.get_type()
                base = self._compile_expr(expr.base, subscripts)
                match exponent:
                    case Int(exp_value):
                        neg = False
                        if exp_value < 0:
                            exp_value = -exp_value
                            neg = True
                        ret = base
                        for _ in range(exp_value - 1):
                            ret = self._compile_mul(ret, base_type, base, base_type)
                        if neg:
                            ret = self._compile_div(self.parent.real_type.from_int(1), base_type, ret, base_type)
                        return ret
                    case Rational(1, 2):
                        return self._block.sqrt(self._compile_expr(expr.base, subscripts))
                    case Rational(-1, 2):
                        return self._compile_div(self.parent.real_type.from_int(1), base_type, self._block.sqrt(self._compile_expr(expr.base, subscripts)), base_type)
                    case _:
                        exp_value = self._compile_expr(exponent, ())
                        return self._block.pow(self._compile_expr(expr.base, subscripts), exp_value)
            case Sin(expr):
                return self._block.sin(self._compile_expr(expr, subscripts))
            case Cos(expr):
                return self._block.cos(self._compile_expr(expr, subscripts))
            case Ln(expr):
                return self._block.ln(self._compile_expr(expr, subscripts))
            case Exp(expr):
                return self._block.exp(self._compile_expr(expr, subscripts))

        raise TypeError(f'unsupported expression: {expr}')

    def _add_symbol(self, expr: Symbol, is_ref: bool) -> ArgInfo:
        jc = self.parent
        assert expr not in self._symbol_values
        type = expr.type
        shape = expr.shape
        assert shape is not None, f"cannot compile symbol {expr} with unspecified shape"
        if len(shape) == 0:
            llvm_type = self.parent.convert_type(type)
            if is_ref:
                ret = ScalarArgInfo(
                    self._fn.add_arg(PointerType(llvm_type)).index,
                    True,
                )
                self._symbol_values[expr] = ret
                return ret
            else:
                ret = ScalarArgInfo(
                    self._fn.add_arg(llvm_type).index,
                    False,
                )
                self._symbol_values[expr] = ret
                return ret
        else:
            for i in shape:
                t = i.get_type()
                assert t == IntegerType(), f"integer type expected for shape, got {t}"
            ret = ArrayArgInfo(
                self._fn.add_arg(PointerType(jc.convert_type(type))).index,
                tuple(self._fn.add_arg(jc.index_type).index for _ in range(len(shape))),
            )
            self._symbol_values[expr] = ret
            self._array_symbol_shapes[expr] = tuple(self._compile_expr(i, ()) for i in shape)
            return ret

    def _compile_lvalue(self, expr: Expr, subscripts: tuple[Value, ...]) -> Value:
        match expr:
            case Symbol():
                sym = self._symbol_values[expr]
                assert sym is not None
                match sym:
                    case ScalarArgInfo():
                        if sym.is_ref:
                            return self._fn.get_arg(sym.value)
                        else:
                            raise TypeError(f"cannot use {expr} as left-value")
                    case ArrayArgInfo():
                        return self._compile_array_symbol_access(sym, subscripts)
        raise ValueError(f"cannot use {expr} as left-value")

    def scan_lvalue_symbols(self, expr: Expr):
        match expr:
            case Symbol():
                self._add_symbol(expr, True)
            case Roll():
                self.scan_lvalue_symbols(expr.expr)
            case Slice():
                self.scan_lvalue_symbols(expr.expr)

    def _compile_expr(self, expr: Expr, subscripts: tuple[Value, ...]) -> Value:
        cache_key = (expr, subscripts)
        if cache_key in self._expr_cache:
            return self._expr_cache[cache_key]
        result = self._compile_expr_no_cache(expr, subscripts)
        self._expr_cache[cache_key] = result
        return result

    def _prepare_tid(self) -> ArgValue:
        if self._tid is None:
            self._tid = self._fn.add_arg(self.parent.index_type, "tid")
        return self._tid

    def _compile_assignment(self, expr: AssignExpr):
        shape = expr.shape
        assert shape is not None, f"cannot compile expression {expr} with unspecified shape"
        shape = tuple(self._compile_expr(i, ()) for i in shape)
        indices = self._compile_unpack_subscripts(shape, self._prepare_tid())
        lhs_ptr = self._compile_lvalue(expr.lhs, indices)
        rhs_value = self._compile_expr(expr.rhs, indices)
        match expr.op:
            case '':
                self._block.store(lhs_ptr, rhs_value)
            case '+':
                self._block.store(lhs_ptr, self._block.add(self._block.load(lhs_ptr), rhs_value))
            case '-':
                self._block.store(lhs_ptr, self._block.sub(self._block.load(lhs_ptr), rhs_value))
            case '*':
                self._block.store(lhs_ptr, self._block.mul(self._block.load(lhs_ptr), rhs_value))
            case '/':
                self._block.store(lhs_ptr, self._block.div(self._block.load(lhs_ptr), rhs_value))
            case _:
                raise ValueError(f"unknown op {expr.op}")

    def compile_assignments(self, exprs: list[AssignExpr]):
        assert not self._finished
        self._finished = True

        # check total size
        first_size = exprs[0].total_size()
        for expr in exprs[1:]:
            expr_size = expr.total_size()
            assert first_size == expr_size, f"incompatible expressions {exprs[0]} and {expr}, with incompatible total sizes {first_size} and {expr_size}"

        for expr in exprs:
            self.scan_lvalue_symbols(expr.lhs)

        assert first_size is not None, "cannot compile expression with unspecified shape"
        total_size_value = self._compile_expr(first_size, ())

        for expr in exprs:
            self._compile_assignment(expr)
        self._block.ret()
        self._fn.set_return_type(VoidType())
        return KernelFunctionIR(self._fn, self._symbol_values, self._tid)

@dataclass
class KernelFunctionIR:
    fn: Function
    args: dict[Symbol, ArgInfo]
    tid: ArgValue | None

    def print(self):
        ret = stringify_symbol_args(self.args)
        if self.tid is not None:
            ret.append(f"tid(%{self.tid.index})")
        return ret

def stringify_symbol_args(args: dict[Symbol, ArgInfo]):
    elems: list[str] = []
    for sym, info in args.items():
        match info:
            case ScalarArgInfo():
                elems.append(('&' if info.is_ref else '') + f"{sym}(%{info.value})")
            case ArrayArgInfo():
                strides = ', '.join(f"%{s}" for s in info.strides)
                elems.append(f"{sym}(%{info.ptr}): Array(strides=({strides}))")
    return elems

class JitCompiler:
    real_type: FloatType
    index_type: IntType

    complex_type: StructType

    def __init__(self, real_type: FloatType = Float64Type(), index_type: IntType = IntType(64, True)):
        self.real_type = real_type
        self.index_type = index_type
        self.complex_type = StructType(real_type, real_type)

    def convert_type(self, type: Expr) -> Type:
        match type:
            case IntegerType():
                return self.index_type
            case RealType():
                return self.real_type
            case ComplexType():
                return self.complex_type
            case _:
                raise ValueError(f"cannot convert type {type}")

    def expand_complex_value(self, b: BasicBlock, value: Value) -> tuple[Value, Value]:
        match value:
            case AggregateValue():
                assert value.type == self.complex_type, f"complex type expected, got {value.type}"
                assert len(value.values) == 2
                return value.values
            case _:
                return b.extract_value(value, 0), b.extract_value(value, 1)

    def complex_add(self, block: BasicBlock, a: Value, b: Value):
        a_re, a_im = self.expand_complex_value(block, a)
        b_re, b_im = self.expand_complex_value(block, b)
        return AggregateValue(self.complex_type, (
            block.add(a_re, b_re),
            block.add(a_im, b_im),
        ))

    def complex_mul(self, block: BasicBlock, a: Value, b: Value):
        a_re, a_im = self.expand_complex_value(block, a)
        b_re, b_im = self.expand_complex_value(block, b)
        return AggregateValue(self.complex_type, (
            block.sub(
                block.mul(a_re, b_re),
                block.mul(a_im, b_im),
            ),
            block.add(
                block.mul(a_re, b_im),
                block.mul(a_im, b_re),
            ),
        ))

    def complex_div(self, block: BasicBlock, a: Value, b: Value):
        b_re, b_im = self.expand_complex_value(block, b)
        den = block.add(
            block.mul(b_re, b_re),
            block.mul(b_im, b_im),
        )
        b_re = block.div(b_re, den)
        b_im = block.fneg(block.div(b_im, den))
        return self.complex_mul(block, a, AggregateValue(self.complex_type, (b_re, b_im)))

    def coerce_to_complex_type(self, block: BasicBlock, value: Value, value_type: Expr):
        match value_type:
            case ComplexType():
                return value
            case RealType():
                return AggregateValue(self.complex_type, (value, FloatValue(0, self.real_type)))
            case IntegerType():
                return AggregateValue(self.complex_type, (block.int_to_float(value, self.real_type), FloatValue(0, self.real_type)))
            case _:
                raise TypeError(f"cannot coerce type {value_type} to complex")

    def coerce_to_real_type(self, block: BasicBlock, value: Value, value_type: Expr):
        match value_type:
            case RealType():
                return value
            case IntegerType():
                return block.int_to_float(value, self.real_type)
            case _:
                raise TypeError(f"cannot coerce type {value_type} to real")

    def coerce(self, block: BasicBlock, value: Value, value_type: Expr, target_type: Expr):
        match target_type:
            case ComplexType():
                return self.coerce_to_complex_type(block, value, value_type)
            case RealType():
                return self.coerce_to_real_type(block, value, value_type)
            case IntegerType():
                if value_type != IntegerType():
                    raise TypeError(f"cannot coerce {value_type} to integer")
                return value
            case _:
                raise TypeError("????")

_TYPE_ORDER: list[type[Expr]] = [ComplexType, RealType, IntegerType]

def combine_types(type1: Expr, type2: Expr):
    for t in _TYPE_ORDER:
        if isinstance(type1, t) or isinstance(type2, t):
            return t()
    raise TypeError(f"cannot combine incompatible types {type1} and {type2} (or some of them is not type)")
