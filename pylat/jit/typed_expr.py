from abc import abstractmethod
from dataclasses import dataclass
from typing import override

from ..expr import Expr, Plus, Power, Roll, Slice, Symbol, Times, UnaryNumericFunction

class Type:
    def is_subtype(self, other: 'Type') -> bool:
        return False

class IntegerType(Type):
    @override
    def is_subtype(self, other: Type) -> bool:
        match other:
            case IntegerType():
                return True
            case _:
                return False

class RealType(Type):
    @override
    def is_subtype(self, other: Type) -> bool:
        match other:
            case IntegerType() | RealType():
                return True
            case _:
                return False

class ComplexType(Type):
    @override
    def is_subtype(self, other: Type) -> bool:
        match other:
            case IntegerType() | RealType() | ComplexType():
                return True
            case _:
                return False

class LowerType:
    @abstractmethod
    def as_type(self) -> Type:
        raise NotImplementedError

@dataclass
class IntType(LowerType):
    bits: int
    signed: bool

    def as_type(self) -> Type:
        return IntegerType()

@dataclass
class FloatType(LowerType):
    bits: int

    def as_type(self) -> Type:
        return RealType()

@dataclass
class ComplexFloatType(LowerType):
    bits: int

    def as_type(self) -> Type:
        return ComplexType()

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
    _symbol_types: dict[Symbol, LowerType]
    _symbol_shapes: dict[Symbol, tuple[Expr, ...]]

    def __init__(self) -> None:
        self._symbol_types = {}
        self._symbol_shapes = {}

    def set_symbol(self, expr: Symbol, type: LowerType | None = None, shape: tuple[Expr, ...] | None = None):
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

    def get_type(self, expr: Expr) -> Type:
        if isinstance(expr, Symbol):
            return self._ctx.get_type(expr).as_type()

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
