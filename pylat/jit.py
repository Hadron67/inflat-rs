from dataclasses import dataclass

from .expr import ComplexType, Cos, Exp, Expr, Float, Int, Ln, Plus, Power, Rational, RealType, Roll, Sin, Slice, Symbol, Times
from .llvm import ArgValue, BasicBlockBuilder, Float64Type, FloatType, Function, IntType, IntValue, Module, PointerType, Type, Value

class ArgInfo:
    pass

@dataclass
class ScalarArgInfo(ArgInfo):
    value: ArgValue
    is_ref: bool

@dataclass
class ArrayArgInfo(ArgInfo):
    ptr: ArgValue
    shape: tuple[Value, ...]
    strides: tuple[ArgValue, ...]

class ModuleBuilder:
    parent: 'JitCompiler'
    mod: Module
    _complex_types: dict[Type, Type]

    def __init__(self, parent: 'JitCompiler', mod: Module) -> None:
        self.sparent = parent
        self.smod = mod
        self._complex_types = {}

    def get_complex_type(self, f: Type) -> Type:
        if f in self._complex_types:
            return self._complex_types[f]
        type = self.mod.add_struct_type(f'struct.complex.{f}', (f, f))
        self._complex_types[f] = type
        return type

    def convert_type(self, type: Expr) -> Type:
        match type:
            case RealType():
                return self.parent.real_type
            case ComplexType():
                return self.get_complex_type(self.parent.real_type)
            case _:
                raise ValueError(f"cannot convert type {type}")

class FunctionCompiler:
    parent: ModuleBuilder
    _expr_cache: dict[tuple[Expr, tuple[Value, ...], Type], Value]
    _subscript_cache: dict[tuple[tuple[Value, ...], tuple[Value, ...]], Value]
    _fn: Function
    _block: BasicBlockBuilder
    _symbol_values: dict[Symbol, ArgInfo]
    _tid: ArgValue | None

    def __init__(self, parent: ModuleBuilder, fn_name: str = 'fn.') -> None:
        self.parent = parent
        self._fn = parent.mod.add_function(fn_name)
        self._block = self._fn.add_block()
        self._expr_cache = {}
        self._symbol_values = {}
        self._tid = None

    def _compile_add(self, left: Value, right: Value) -> Value:
        # TODO: complex
        return self._block.add(left, right)

    def _compile_mul(self, left: Value, right: Value) -> Value:
        # TODO: ditto
        return self._block.mul(left, right)

    def _compile_div(self, left: Value, right: Value) -> Value:
        # TODO: ditto
        return self._block.div(left, right)

    def _compile_unpack_subscripts(self, sizes: tuple[Value, ...], packed: Value) -> tuple[Value, ...]:
        assert len(sizes) > 0
        ret: list[Value] = []
        for size in sizes[-1:0:-1]:
            ret.append(self._block.rem(packed, size))
            packed = self._block.div(packed, size)
        return tuple(ret[-1::-1])

    def _compile_subscript_no_cache(self, strides: tuple[Value, ...], subscripts: tuple[Value, ...]) -> Value:
        assert len(subscripts) >= len(strides)
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
        return self._block.get_array_element_ptr(info.ptr, self._compile_subscript(info.strides, subscripts))

    def _compile_expr_no_cache(self, expr: Expr, subscripts: tuple[Value, ...], type: Type) -> Value:
        jc = self.parent.parent

        match expr:
            case Int(value):
                return type.from_int(value)
            case Rational(numerator, denominator):
                return type.from_float(numerator / denominator)
            case Float(value):
                return type.from_float(value)
            case Symbol():
                sym = self._symbol_values[expr]
                if sym is None:
                    sym = self._add_symbol(expr, False)
                match sym:
                    case ScalarArgInfo():
                        if sym.is_ref:
                            return self._block.coerce(self._block.load(sym.value), type)
                        else:
                            return self._block.coerce(sym.value, type)
                    case ArrayArgInfo():
                        ret = self._block.load(self._compile_array_symbol_access(sym, subscripts))
                        return self._block.coerce(ret, type)
            case Roll():
                expr_shape = expr.expr.get_shape()
                assert expr_shape is not None, "cannot compile unspecified shape"
                len = self._compile_expr(expr_shape[expr.axis], (), jc.index_type)
                new_index = subscripts[expr.axis]
                new_index = self._block.add(new_index, IntValue(-expr.amount, jc.index_type))
                new_index = self._block.rem(new_index, len)
                subscripts = subscripts[:expr.axis] + (new_index,) + subscripts[expr.axis + 1:]
                return self._compile_expr(expr.expr, subscripts, type)
            case Slice():
                return self._compile_expr(expr.expr, subscripts[:expr.axis] + (IntValue(expr.index, jc.index_type),) + subscripts[expr.axis + 1:], type)
            case Plus(children):
                ret = self._compile_expr(children[0], subscripts, type)
                for child in children[1:]:
                    ret = self._compile_add(ret, self._compile_expr(child, subscripts, type))
                return ret
            case Times(children):
                ret = self._compile_expr(children[0], subscripts, type)
                for child in children[1:]:
                    ret = self._compile_mul(ret, self._compile_expr(child, subscripts, type))
                return ret
            case Power(_, exponent):
                base = self._compile_expr(expr.base, subscripts, type)
                base_type = base.get_type()
                match exponent:
                    case Int(exp_value):
                        neg = False
                        if exp_value < 0:
                            exp_value = -exp_value
                            neg = True
                        ret = base
                        for _ in range(exp_value - 1):
                            ret = self._compile_mul(ret, base)
                        if neg:
                            ret = self._compile_div(base_type.from_int(1), ret)
                        return ret
                    case Rational(1, 2):
                        return self._block.sqrt(self._compile_expr(expr.base, subscripts, type))
                    case Rational(-1, 2):
                        return self._compile_div(base_type.from_int(1), self._block.sqrt(self._compile_expr(expr.base, subscripts, type)))
                    case _:
                        exp_value = self._compile_expr(exponent, (), type)
                        return self._block.pow(self._compile_expr(expr.base, subscripts, type), exp_value)
            case Sin(expr):
                return self._block.sin(self._compile_expr(expr, subscripts, type))
            case Cos(expr):
                return self._block.cos(self._compile_expr(expr, subscripts, type))
            case Ln(expr):
                return self._block.ln(self._compile_expr(expr, subscripts, type))
            case Exp(expr):
                return self._block.exp(self._compile_expr(expr, subscripts, type))

        raise TypeError(f'unsupported expression: {expr}')

    def _add_symbol(self, expr: Symbol, is_ref: bool) -> ArgInfo:
        assert expr not in self._symbol_values
        type = expr.type
        shape = expr.shape
        assert shape is not None, f"cannot compile symbol {expr} with unspecified shape"
        if len(shape) == 0:
            llvm_type = self.parent.convert_type(type)
            if is_ref:
                ret = ScalarArgInfo(
                    self._fn.add_arg(PointerType(llvm_type)),
                    True,
                )
                self._symbol_values[expr] = ret
                return ret
            else:
                ret = ScalarArgInfo(
                    self._fn.add_arg(llvm_type),
                    False,
                )
                self._symbol_values[expr] = ret
                return ret
        else:
            ret = ArrayArgInfo(
                self._fn.add_arg(PointerType(self.parent.convert_type(type))),
                tuple(self._compile_expr(i, (), self.parent.parent.index_type) for i in shape),
                tuple(self._fn.add_arg(self.parent.parent.index_type) for _ in range(len(shape))),
            )
            self._symbol_values[expr] = ret
            return ret


    def _compile_lvalue(self, expr: Expr, subscripts: tuple[Value, ...]) -> Value:
        match expr:
            case Symbol():
                sym = self._symbol_values[expr]
                assert sym is not None
                match sym:
                    case ScalarArgInfo():
                        if sym.is_ref:
                            return sym.value
                        else:
                            raise TypeError(f"cannot use {expr} as left-value")
                    case ArrayArgInfo():
                        return self._compile_array_symbol_access(sym, subscripts)
        raise ValueError(f"cannot use {expr} as left-value")

    def _scan_lvalue_symbols(self, expr: Expr):
        match expr:
            case Symbol():
                self._add_symbol(expr, True)
            case Roll():
                self._scan_lvalue_symbols(expr.expr)
            case Slice():
                self._scan_lvalue_symbols(expr.expr)

    def _compile_expr(self, expr: Expr, subscripts: tuple[Value, ...], type: Type) -> Value:
        cache_key = (expr, subscripts, type)
        if cache_key in self._expr_cache:
            return self._block.coerce(self._expr_cache[cache_key], type)
        result = self._compile_expr_no_cache(expr, subscripts, type)
        self._expr_cache[cache_key] = result
        return result

    def _prepare_tid(self) -> ArgValue:
        if self._tid is None:
            self._tid = self._fn.add_arg(self.parent.parent.index_type, "tid")
        return self._tid

    def _compile_assignment(self, lhs: Expr, rhs: Expr):
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert lhs_type.is_subtype(rhs_type), f"cannot assign type {rhs_type} to {lhs_type}"
        lhs_shape = lhs.get_shape()
        assert lhs_shape is not None, f"cannot compile expression {lhs} with unspecified shape"
        shape = tuple(self._compile_expr(i, (), self.parent.parent.index_type) for i in lhs_shape)
        indices = self._compile_unpack_subscripts(shape, self._prepare_tid())
        llvm_type = self.parent.convert_type(lhs_type)
        lhs_ptr = self._compile_lvalue(lhs, indices)
        rhs_value = self._compile_expr(rhs, indices, llvm_type)
        self._block.store(lhs_ptr, rhs_value)

class JitCompiler:
    real_type: FloatType
    index_type: IntType

    def __init__(self, real_type: FloatType = Float64Type(), index_type: IntType = IntType(64, True)):
        self.real_type = real_type
        self.index_type = index_type
