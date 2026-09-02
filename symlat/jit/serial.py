from typing import override

from .backend import Backend, LoopKernel, ReductionKernel
from .llvm import BasicBlock, IntValue, Value
from .util import ForLoopBuilder


class SerialBackend(Backend):
    """A backend that runs the loop inline on the calling thread.

    Useful for small loop sizes, where forking a parallel region would cost
    more than running the loop itself.
    """

    @override
    def compile_paralell_loop(self, block: BasicBlock, args: tuple[Value, ...], kernel: LoopKernel, reduction: ReductionKernel | None = None, reduction_ptr: Value | None = None) -> BasicBlock:
        """Emit one serial loop over ``[0, total_size)`` at ``block``.

        Returns the block where execution resumes once the loop has finished.
        The comparison is signed so that an empty loop (total size 0, whose
        upper bound wraps to -1) does not run at all.
        """
        index_type = kernel.get_index_type()
        b, total_size = kernel.compile_total_size(block, args)
        loop_builder = ForLoopBuilder(
            b, True,
            IntValue(0, index_type),
            b.sub(total_size, IntValue(1, index_type)),
            IntValue(1, index_type),
        )
        b = loop_builder.body_entry
        b, value = kernel.compile_body(b, args, loop_builder.loop_var, None)
        if reduction is not None:
            assert reduction_ptr is not None
            # single threaded: accumulate straight into the shared accumulator
            b = reduction.reduce(b, reduction_ptr, value)
        return loop_builder.end(b)
