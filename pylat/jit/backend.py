

from abc import abstractmethod

from .llvm import BasicBlock, IFunction, IntType, Type, Value

class CompiledBackendFunction:
    @abstractmethod
    def call(self, *args):
        raise NotImplementedError

    @abstractmethod
    def print_all(self) -> list[str]:
        raise NotImplementedError

class ParallelForLoopProvider(IFunction):
    """
        An abstract for loop provider. It builds a function roughly like

        fn run_loop(...) {
            ... <prologue> ...
            #pragma parallel for
            for (i = 0; i <= size; i += 1) {
                ... <body> ...
            }
        }

    """
    @abstractmethod
    def begin_prologue(self) -> BasicBlock:
        raise NotImplementedError

    @abstractmethod
    def begin_loop(self, prologue_end: BasicBlock, total_size: Value) -> tuple[BasicBlock, Value]:
        """
            To be called before compiling the kernel.

            Returns the loop variable
        """
        raise NotImplementedError

    @abstractmethod
    def end(self, block: BasicBlock) -> CompiledBackendFunction:
        raise NotImplementedError

class LoopKernel:
    @abstractmethod
    def get_index_type(self) -> IntType:
        raise NotImplementedError

    @abstractmethod
    def get_args(self) -> tuple[Type, ...]:
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
