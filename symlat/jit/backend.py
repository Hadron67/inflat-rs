from abc import abstractmethod

from .helper import MaybeComplexValue
from .llvm import BasicBlock, IntType, Ordering, Value
from .type import LowerType


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
    def get_type(self) -> LowerType:
        raise NotImplementedError

    @abstractmethod
    def store_initial_value(self, block: BasicBlock, value_ptr: Value):
        raise NotImplementedError

    @abstractmethod
    def reduce(self, block: BasicBlock, acc_ptr: Value, value: MaybeComplexValue, ordering: Ordering | None = None) -> BasicBlock:
        raise NotImplementedError

class Backend:
    """A backend that can emit one parallel loop into an existing function.

    The enclosing (external) function is owned by the caller: ``compile.py``
    declares its arguments, feeds their entry block to
    :meth:`compile_paralell_loop` and JIT-compiles the finished function.  A
    backend only knows how to generate the loop itself.
    """

    @abstractmethod
    def compile_paralell_loop(self, block: BasicBlock, args: tuple[Value, ...], kernel: LoopKernel, reduction: ReductionKernel | None = None, reduction_ptr: Value | None = None) -> BasicBlock:
        """Emit one parallel loop at ``block`` of the current function.

        ``args`` holds the values of the enclosing function's arguments, in the
        order of ``kernel.get_args()``.  When ``reduction`` is given,
        ``reduction_ptr`` is the caller-owned accumulator the parallel region
        reduces into; it already holds the initial value.  Returns the block
        where execution resumes once the loop has finished.
        """
        raise NotImplementedError
