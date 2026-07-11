

from abc import abstractmethod
import ctypes

from pylat.jit.argpass import LowerType
from pylat.jit.helper import MaybeComplexValue

from .llvm import BasicBlock, IntType, Ordering, Type, Value

class CompiledBackendFunction:
    @abstractmethod
    def call(self, *args: ctypes._CDataType):
        raise NotImplementedError

    @abstractmethod
    def print_all(self) -> list[str]:
        raise NotImplementedError

class DebugInterface:
    @abstractmethod
    def echo(self, block: BasicBlock, *args: Value | str):
        raise NotImplementedError

class LoopKernel:
    @abstractmethod
    def get_index_type(self) -> IntType:
        raise NotImplementedError

    @abstractmethod
    def get_args(self) -> tuple[LowerType, ...]:
        raise NotImplementedError

    @abstractmethod
    def compile_total_size(self, begin: BasicBlock, args: tuple[Value, ...]) -> tuple[BasicBlock, Value]:
        raise NotImplementedError

    @abstractmethod
    def compile_body(self, begin: BasicBlock, args: tuple[Value, ...], loop_var: Value, debug: DebugInterface) -> tuple[BasicBlock, MaybeComplexValue]:
        raise NotImplementedError

class ReductionKernel:
    @abstractmethod
    def get_type(self) -> Type:
        raise NotImplementedError

    @abstractmethod
    def store_initial_value(self, block: BasicBlock, value_ptr: Value):
        raise NotImplementedError

    @abstractmethod
    def reduce(self, block: BasicBlock, acc_ptr: Value, value: MaybeComplexValue, ordering: Ordering | None = None) -> BasicBlock:
        raise NotImplementedError

class Backend:
    @abstractmethod
    def compile_paralell_loop(self, kernel: LoopKernel, reduction: ReductionKernel | None = None) -> CompiledBackendFunction:
        raise NotImplementedError
