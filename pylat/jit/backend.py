

from abc import abstractmethod
import ctypes

from pylat.jit.argpass import LowerType

from .llvm import BasicBlock, IntType, Value

class CompiledBackendFunction:
    @abstractmethod
    def call(self, *args: ctypes._CDataType):
        raise NotImplementedError

    @abstractmethod
    def print_all(self) -> list[str]:
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
    def compile_body(self, begin: BasicBlock, args: tuple[Value, ...], loop_var: Value) -> BasicBlock:
        raise NotImplementedError

class Backend:
    @abstractmethod
    def compile_paralell_loop(self, kernel: LoopKernel) -> CompiledBackendFunction:
        raise NotImplementedError
