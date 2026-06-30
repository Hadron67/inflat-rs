from abc import abstractmethod
import ctypes
from dataclasses import dataclass
from typing import Any, Callable, override

from ..expr import Expr
from .llvm import FloatType, IntType

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
