

from abc import abstractmethod

from .llvm import BasicBlock, IFunction, IntType, Value

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
    def begin_loop(self, index_type: IntType, prologue_end: BasicBlock, total_size: Value) -> tuple[BasicBlock, Value]:
        """
            To be called before compiling the kernel.

            Returns the loop variable
        """
        raise NotImplementedError

    @abstractmethod
    def end(self, block: BasicBlock) -> CompiledBackendFunction:
        raise NotImplementedError

class Backend:
    @abstractmethod
    def get_for_loop_provider(self) -> ParallelForLoopProvider:
        raise NotImplementedError
