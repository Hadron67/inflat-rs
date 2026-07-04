from abc import abstractmethod
import ctypes
from dataclasses import dataclass
from inspect import ArgInfo
from typing import Any, Callable, override

from . import llvm

from ..expr import AssignExpr, Expr, Plus, Power, Roll, Slice, Symbol, Times, UnaryNumericFunction

class Type:
    def is_subtype(self, other: 'Type') -> bool:
        return False

@dataclass
class IntegerType(Type):
    @override
    def is_subtype(self, other: Type) -> bool:
        match other:
            case IntegerType():
                return True
            case _:
                return False

@dataclass
class RealType(Type):
    @override
    def is_subtype(self, other: Type) -> bool:
        match other:
            case IntegerType() | RealType():
                return True
            case _:
                return False

@dataclass
class ComplexType(Type):
    @override
    def is_subtype(self, other: Type) -> bool:
        match other:
            case IntegerType() | RealType() | ComplexType():
                return True
            case _:
                return False

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
    def to_llvm_type(self) -> llvm.Type:
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
    def to_llvm_type(self) -> llvm.Type:
        match self.bits:
            case 32:
                return llvm.FloatType(32)
            case 64:
                return llvm.FloatType(64)
            case _:
                raise ValueError(f"unsupported bit size: {self.bits}")

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

def peer_type(types: tuple[Type, ...]) -> Type:
    assert len(types) > 0
    if len(types) == 1:
        return types[0]
    ret = types[0]
    for t in types[1:]:
        if not ret.is_subtype(t):
            assert t.is_subtype(ret), f"incompatible types {t} and {ret}"
            ret = t
    return ret

class TypeContext:
    _symbol_types: dict[Symbol, tuple[Type, LowerType]]
    _symbol_shapes: dict[Symbol, tuple[Expr, ...]]

    def __init__(self) -> None:
        self._symbol_types = {}
        self._symbol_shapes = {}

    def set_symbol(self, expr: Symbol, type: tuple[Type, LowerType] | None = None, shape: tuple[Expr, ...] | None = None):
        if type is not None:
            assert expr not in self._symbol_types
            self._symbol_types[expr] = type
        if shape is not None:
            assert expr not in self._symbol_shapes
            self._symbol_shapes[expr] = shape

    def get_type(self, expr: Symbol):
        return self._symbol_types[expr]

    def get_shape(self, expr: Symbol):
        return self._symbol_shapes[expr]

class TypeCache:
    _ctx: TypeContext
    _type_cache: dict[Expr, Type]
    _shape_cache: dict[Expr, tuple[Expr, ...]]

    def __init__(self, ctx: TypeContext) -> None:
        self._ctx = ctx
        self._type_cache = {}
        self._shape_cache = {}

    def get_symbol_type(self, expr: Symbol):
        return self._ctx.get_type(expr)

    def get_symbol_shape(self, expr: Symbol):
        return self._ctx.get_shape(expr)

    def get_type(self, expr: Expr) -> Type:
        if isinstance(expr, Symbol):
            return self._ctx.get_type(expr)[0]

        if expr in self._type_cache:
            return self._type_cache[expr]
        type = self._get_type_no_cache(expr)
        self._type_cache[expr] = type
        return type

    def get_shape(self, expr: Expr):
        if isinstance(expr, Symbol):
            return self._ctx.get_shape(expr)

        if expr in self._shape_cache:
            return self._shape_cache[expr]
        ret = self._get_shape_no_cache(expr)
        self._shape_cache[expr] = ret
        return ret

    def _get_type_no_cache(self, expr: Expr) -> Type:
        match expr:
            case Plus(children) | Times(children):
                return peer_type(tuple(self.get_type(a) for a in children))
            case Power(base, exp):
                return self.get_type(base)
            case Roll() | Slice():
                return self.get_type(expr.expr)
            case UnaryNumericFunction():
                type = self.get_type(expr.expr)
                if isinstance(type, ComplexType):
                    return type
                return RealType()
            case _:
                raise TypeError(f"cannot get type from {expr}")

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
    type: Type
    shape: tuple[Expr, ...]

    def __init__(self, expr: AssignExpr, ctx: TypeCache) -> None:
        self.expr = expr
        lhs_type = ctx.get_type(expr.lhs)
        rhs_type = ctx.get_type(expr.rhs)
        assert lhs_type.is_subtype(rhs_type), f"cannot assign type {rhs_type} to {lhs_type}"
        self.type = lhs_type

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
