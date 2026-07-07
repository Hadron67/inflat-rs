from abc import abstractmethod
import ctypes
from dataclasses import dataclass
from typing import Any, Callable, override

from . import llvm

from ..expr import AssignExpr, Expr, Plus, Power, Roll, Slice, Symbol, SymbolShape, Times, UnaryNumericFunction

class LowerType:
    @staticmethod
    def from_numpy_dtype(dtype: str) -> 'LowerType':
        match dtype:
            case 'int8':
                return IntType(8, True)
            case 'uint8':
                return IntType(8, False)
            case 'int16':
                return IntType(16, True)
            case 'uint16':
                return IntType(16, False)
            case 'int32':
                return IntType(32, True)
            case 'uint32':
                return IntType(32, False)
            case 'int64':
                return IntType(64, True)
            case 'uint64':
                return IntType(64, False)
            case 'float32':
                return FloatType(32)
            case 'float64':
                return FloatType(64)
            case 'complex64':
                return ComplexFloatType(FloatType(32))
            case 'complex128':
                return ComplexFloatType(FloatType(64))
            case _:
                raise ValueError(f"unsupported dtype: {dtype}")

    @abstractmethod
    def to_ctype(self) -> type[ctypes._CDataType]:
        raise NotImplementedError

    @abstractmethod
    def to_llvm_type(self) -> llvm.Type:
        raise NotImplementedError

@dataclass(frozen=True)
class IntType(LowerType):
    bits: int
    signed: bool

    @staticmethod
    def range_of(bits: int, signed: bool):
        if signed:
            return (-2**(bits - 1), 2**(bits - 1) - 1)
        else:
            return (0, 2**bits - 1)

    def range(self):
        return IntType.range_of(self.bits, self.signed)

    @override
    def to_ctype(self) -> type[ctypes._CDataType]:
        match self.bits:
            case 8:
                return ctypes.c_int8 if self.signed else ctypes.c_uint8
            case 16:
                return ctypes.c_int16 if self.signed else ctypes.c_uint16
            case 32:
                return ctypes.c_int if self.signed else ctypes.c_uint
            case 64:
                return ctypes.c_long if self.signed else ctypes.c_ulong
            case _:
                raise ValueError(f"unsupported bit size: {self.bits}")

    @override
    def to_llvm_type(self) -> llvm.IntType:
        match self.bits:
            case 8:
                return llvm.IntType(8)
            case 16:
                return llvm.IntType(16)
            case 32:
                return llvm.IntType(32)
            case 64:
                return llvm.IntType(64)
            case _:
                raise ValueError(f"unsupported bit size: {self.bits}")

    def to_llvm_value(self, value: int) -> llvm.Value:
        return llvm.IntValue(value, self.to_llvm_type())

@dataclass(frozen=True)
class FloatType(LowerType):
    bits: int

    @override
    def to_ctype(self) -> type[ctypes._CDataType]:
        match self.bits:
            case 32:
                return ctypes.c_float
            case 64:
                return ctypes.c_double
            case _:
                raise ValueError(f"unsupported bit size: {self.bits}")

    @override
    def to_llvm_type(self) -> llvm.FloatType:
        match self.bits:
            case 32:
                return llvm.FloatType(32)
            case 64:
                return llvm.FloatType(64)
            case _:
                raise ValueError(f"unsupported bit size: {self.bits}")

    def to_llvm_value(self, value: float) -> llvm.Value:
        return llvm.FloatValue(value, self.to_llvm_type())

_LLVM_COMPLEX_FLOAT32 = llvm.StructType(llvm.FloatType(32), llvm.FloatType(32))
_LLVM_COMPLEX_FLOAT64 = llvm.StructType(llvm.FloatType(64), llvm.FloatType(64))

@dataclass(frozen=True)
class ComplexFloatType(LowerType):
    type: FloatType

    @override
    def to_ctype(self) -> type[ctypes._CDataType]:
        match self.type.bits:
            case 32:
                return ctypes.c_float_complex
            case 64:
                return ctypes.c_double_complex
            case _:
                raise ValueError(f"unsupported bit size: {self.type.bits}")

    @override
    def to_llvm_type(self) -> llvm.Type:
        match self.type.bits:
            case 32:
                return _LLVM_COMPLEX_FLOAT32
            case 64:
                return _LLVM_COMPLEX_FLOAT64
            case _:
                raise ValueError(f"unsupported bit size: {self.type.bits}")

@dataclass(frozen=True)
class PointerType(LowerType):
    type: LowerType

    @override
    def to_ctype(self) -> type[ctypes._CDataType]:
        return ctypes.POINTER(self.type.to_ctype())

    @override
    def to_llvm_type(self) -> llvm.Type:
        return llvm.PointerType(self.type.to_llvm_type())

def merge_shape(shape1: tuple[Expr, ...], shape2: tuple[Expr, ...]) -> tuple[Expr, ...]:
    if len(shape1) == len(shape2):
        assert shape1 == shape2, f"shape mismatch {shape1} != {shape2}"
        return shape1
    ret: list[Expr] = []
    for i in range(max(len(shape1), len(shape2))):
        if i < len(shape1) and i < len(shape2):
            s1 = shape1[-1 - i]
            s2 = shape2[-1 - i]
            assert s1 == s2, f"incompatible shapes {shape1} != {shape2}"
            ret.append(s1)
        elif i < len(shape1):
            ret.append(shape1[-1 - i])
        elif i < len(shape2):
            ret.append(shape2[-1 - i])
    ret.reverse()
    return tuple(ret)

def merge_shapes(*shapes: tuple[Expr, ...]) -> tuple[Expr, ...]:
    assert len(shapes) > 0
    ret = shapes[0]
    for shape in shapes[1:]:
        ret = merge_shape(ret, shape)
    return ret

def _get_complex_peer_type(type: ComplexFloatType, other: LowerType) -> ComplexFloatType:
    match other:
        case ComplexFloatType():
            return ComplexFloatType(_get_float_peer_type(type.type, other.type))
        case FloatType():
            return ComplexFloatType(_get_float_peer_type(type.type, other))
        case IntType():
            return ComplexFloatType(_get_float_peer_type(type.type, other))
        case _:
            raise ValueError(f"unsupported type: {other}")

def _get_float_peer_type(type: FloatType, other: LowerType) -> FloatType:
    if isinstance(other, FloatType):
        return FloatType(max(type.bits, other.bits))
    else:
        return FloatType(type.bits)

def _get_int_peer_type(type: IntType, other: LowerType) -> IntType:
    if isinstance(other, IntType):
        return IntType(max(type.bits, other.bits), other.signed or type.signed)
    raise ValueError(f"unsupported type: {other}")

def get_peer_type(type: LowerType, other: LowerType) -> LowerType:
    if isinstance(type, ComplexFloatType):
        return _get_complex_peer_type(type, other)
    if isinstance(other, ComplexFloatType):
        return _get_complex_peer_type(other, type)
    if isinstance(type, FloatType):
        return _get_float_peer_type(type, other)
    if isinstance(other, FloatType):
        return _get_float_peer_type(other, type)
    if isinstance(type, IntType):
        return _get_int_peer_type(type, other)
    if isinstance(other, IntType):
        return _get_int_peer_type(other, type)
    raise ValueError(f"unsupported type: {type} and {other}")

def get_peer_types(*types: LowerType) -> LowerType:
    if len(types) == 0:
        raise ValueError("no types provided")
    ret = types[0]
    for t in types[1:]:
        ret = get_peer_type(ret, t)
    return ret

class TypeContext:
    _symbol_types: dict[Symbol, LowerType]
    _symbol_shapes: dict[Symbol, tuple[Expr, ...]]
    _symbol_dimension: dict[Symbol, int]

    def __init__(self) -> None:
        self._symbol_types = {}
        self._symbol_shapes = {}
        self._symbol_dimension = {}

    def set_symbol(self, expr: Symbol, type: LowerType | None = None, shape: tuple[Expr, ...] | None = None, dimension: int | None = None):
        if type is not None:
            assert expr not in self._symbol_types
            self._symbol_types[expr] = type
        if shape is not None:
            assert expr not in self._symbol_shapes
            self._symbol_shapes[expr] = shape
        if dimension is not None:
            assert expr not in self._symbol_dimension
            self._symbol_dimension[expr] = dimension

    def get_type(self, expr: Symbol):
        return self._symbol_types[expr]

    def get_shape(self, expr: Symbol):
        return self._symbol_shapes[expr]

    def get_dimension(self, expr: Symbol):
        return self._symbol_dimension[expr]

class TypeCache:
    _ctx: TypeContext
    _shape_cache: dict[Expr, tuple[Expr, ...]]
    _resolved_shapes: dict[SymbolShape, Expr]

    def __init__(self, ctx: TypeContext) -> None:
        self._ctx = ctx
        self._type_cache = {}
        self._shape_cache = {}
        self._resolved_shapes = {}

    def get_symbol_type(self, expr: Symbol):
        return self._ctx.get_type(expr)

    def get_symbol_shape(self, expr: Symbol):
        return self._ctx.get_shape(expr)

    def get_shape(self, expr: Expr):
        if isinstance(expr, Symbol):
            return self._ctx.get_shape(expr)

        if expr in self._shape_cache:
            return self._shape_cache[expr]
        ret = self._get_shape_no_cache(expr)
        self._shape_cache[expr] = ret
        return ret

    def _get_shape_no_cache(self, expr: Expr) -> tuple[Expr, ...]:
        match expr:
            case Plus(children) | Times(children):
                return merge_shapes(*tuple(self.get_shape(a) for a in children))
            case Power(base, exp):
                return merge_shapes(self.get_shape(base), self.get_shape(exp))
            case UnaryNumericFunction():
                return self.get_shape(expr.expr)
            case Roll():
                return self.get_shape(expr.expr)
            case Slice():
                shape = self.get_shape(expr.expr)
                if expr.axis >= len(shape):
                    raise IndexError(f"Axis {expr.axis} is out of bounds for shape {shape}")
                return shape[:expr.axis] + shape[expr.axis + 1:]
            case _:
                raise TypeError(f"cannot get shape from {expr}")

class TypedAssignExpr:
    expr: AssignExpr
    shape: tuple[Expr, ...]

    def __init__(self, expr: AssignExpr, ctx: TypeCache) -> None:
        self.expr = expr

        lhs_shape = ctx.get_shape(expr.lhs)
        rhs_shape = ctx.get_shape(expr.rhs)
        shape = merge_shape(lhs_shape, rhs_shape)
        assert shape == lhs_shape, "incompatible shape"
        self.shape = shape

    def total_size(self):
        return Times.make(self.shape).evaluate()

@dataclass
class TypesConfig:
    real_type: FloatType
    index_type: IntType

class SymbolArgInfo:
    @abstractmethod
    def map_arg(self, op: Callable[[int], int]) -> 'SymbolArgInfo':
        raise NotImplementedError

    @abstractmethod
    def write_one_arg(self, arg_value: Any, args: list[ctypes._CDataType | None], config: TypesConfig):
        raise NotImplementedError

@dataclass
class ScalarArgInfo(SymbolArgInfo):
    value: int
    is_ref: bool

    @override
    def __str__(self) -> str:
        return f"%{self.value}: {'&' if self.is_ref else ''}Scalar"

    @override
    def map_arg(self, op: Callable[[int], int]) -> SymbolArgInfo:
        return ScalarArgInfo(op(self.value), self.is_ref)

@dataclass
class ArrayArgInfo(SymbolArgInfo):
    ptr: int
    strides: tuple[int, ...]

    @override
    def __str__(self) -> str:
        return f"%{self.ptr}: Array(strides=({", ".join(str(i) for i in self.strides)}))"

    @override
    def map_arg(self, op: Callable[[int], int]) -> 'SymbolArgInfo':
        return ArrayArgInfo(op(self.ptr), tuple(op(i) for i in self.strides))
