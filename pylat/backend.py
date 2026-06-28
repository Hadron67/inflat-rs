

from abc import abstractmethod

from .llvm import BasicBlock, Function, Type, Value

class ParallelForLoopProvider:
    @abstractmethod
    def begin(self, index_type: Type, fn: Function, block: BasicBlock, total_size: Value) -> Value:
        """
            To be called before compiling the kernel.

            Returns the loop variable
        """
        raise NotImplementedError
