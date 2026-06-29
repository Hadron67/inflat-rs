from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, override

from .expr import Symbol

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

    @override
    def __str__(self) -> str:
        return f"%{self.ptr}: Array(strides=({", ".join(str(i) for i in self.strides)}))"

    @override
    def map_arg(self, op: Callable[[int], int]) -> 'ArgInfo':
        return ArrayArgInfo(op(self.ptr), tuple(op(i) for i in self.strides))

class SymbolScope:
    _symbol_values: dict[Symbol, ArgInfo]

    def __init__(self) -> None:
        self._symbol_values = {}

    def get_symbol(self, symbol: Symbol):
        return self._symbol_values.get(symbol, None)

    def add_symbol(self, symbol: Symbol, info: ArgInfo):
        assert symbol not in self._symbol_values
        self._symbol_values[symbol] = info

    def items(self):
        return self._symbol_values.items()

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
