from ctypes import CDLL
from typing import override

from .backend import (
    Backend,
    DebugInterface,
    LoopKernel,
    ReductionKernel,
)
from .helper import echo
from .llvm import (
    I8,
    I32,
    I64,
    ArrayType,
    BasicBlock,
    DeclareFunction,
    FnType,
    Function,
    GlobalAggregateValue,
    GlobalStringValue,
    GlobalValueFlags,
    GlobalZeroAggregateValue,
    IcmpOp,
    IntType,
    IntValue,
    NullValue,
    Ordering,
    PointerType,
    StructType,
    Value,
    VoidType,
    VoidValue,
    fn_type,
)
from .util import ForLoopBuilder

_IDEN_T = StructType(
    I32,
    I32,
    I32,
    I32,
    PointerType(I8),
)

_KMPC_CRITICAL_NAME = ArrayType(I32, 8)

def _for_static_init(type: IntType):
    fn_type = FnType((
        PointerType(_IDEN_T), # loc
        I32, # gitd
        I32, # schedtype
        PointerType(I32), # plastiter
        PointerType(type), # plower
        PointerType(type), # pupper
        PointerType(type), # pstride
        type, # incr
        type, # chunk
    ), VoidType())
    return DeclareFunction(f'__kmpc_for_static_init_{type.bits // 8}', fn_type)

_KMPC_FORK_CALL_CALLBACK_TYPE = fn_type(None, PointerType(I32), PointerType(I32), ...)
_KMPC_FORK_CALL_TYPE = fn_type(None, PointerType(_IDEN_T), I32, PointerType(_KMPC_FORK_CALL_CALLBACK_TYPE), ...)

_SIZE_T = I64

_REDUCE_FN = fn_type(None, PointerType(I8), PointerType(I8))
_KMPC_FORK_CALL = DeclareFunction('__kmpc_fork_call', _KMPC_FORK_CALL_TYPE)
_KMPC_FOR_STATIC_FINI = DeclareFunction('__kmpc_for_static_fini', fn_type(None, PointerType(_IDEN_T), I32))
_KMPC_CRITIAL = DeclareFunction('__kmpc_critical', fn_type(None, PointerType(_IDEN_T), I32, PointerType(_KMPC_CRITICAL_NAME)))
_KMPC_END_CRITICAL = DeclareFunction('__kmpc_end_critical', fn_type(None, PointerType(_IDEN_T), I32, PointerType(_KMPC_CRITICAL_NAME)))
_KMPC_BARRIER = DeclareFunction('__kmpc_barrier', fn_type(None, PointerType(_IDEN_T), I32))
_KMPC_REDUCE_NOWAIT = DeclareFunction('__kmpc_reduce_nowait', fn_type(I32, PointerType(_IDEN_T), I32, I32, _SIZE_T, PointerType(I8), PointerType(_REDUCE_FN), PointerType(_KMPC_CRITICAL_NAME)))
_KMPC_END_REDUCE_NOWAIT = DeclareFunction('__kmpc_end_reduce_nowait', fn_type(None, PointerType(_IDEN_T), I32, PointerType(_KMPC_CRITICAL_NAME)))

class _DebugInterface(DebugInterface):
    gtid: Value

    @override
    def __init__(self, gtid: Value) -> None:
        self.gtid = gtid

    @override
    def echo(self, block: BasicBlock, *args: Value | str):
        _echo_sync(block, self.gtid, "[gtid = ", self.gtid, "]", *args)

class OpenMPBackend(Backend):
    def __init__(self, libomp: str | CDLL | None = None) -> None:
        pass

    @override
    def compile_paralell_loop(self, block: BasicBlock, args: tuple[Value, ...], kernel: LoopKernel, reduction: ReductionKernel | None = None, reduction_ptr: Value | None = None) -> BasicBlock:
        """Emit one OpenMP parallel loop at ``block`` of the current function.

        The enclosing function (and its arguments, ``args``) is owned by the
        caller: this method only packs a closure holding the arguments, the
        total loop size and (with a reduction) the accumulator, forks an
        outlined microtask that runs the loop body, and returns the block where
        execution resumes once the parallel region has finished.
        """
        index_type = kernel.get_index_type()
        arg_lower_types = kernel.get_args()
        arg_llvm_types = tuple(a.to_llvm_type() for a in arg_lower_types)
        ident = GlobalAggregateValue(_IDEN_T,
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            GlobalStringValue(b';unknown;unknown;0;0;;\00'),
        )
        closure_type = StructType(*arg_llvm_types, index_type)
        reduction_llvm_type = None
        if reduction is not None:
            reduction_llvm_type = reduction.get_type().to_llvm_type()
            closure_type.add_field(PointerType(reduction_llvm_type))

        # the outlined microtask: __kmpc_fork_call schedules it on every thread
        inner_fn = Function()
        inner_fn.add_args(PointerType(I32), PointerType(I32))
        inner_fn.set_return_type(VoidType(), True)
        inner_fn.add_args(PointerType(closure_type))

        kmpc_for_static_init = _for_static_init(index_type)

        # pack the closure and fork the parallel region at the current block
        b = block
        closure_ptr = b.alloca(closure_type)
        b, total_size = kernel.compile_total_size(b, args)
        cursor = 0
        for value in args:
            b.store(b.get_element_ptr(closure_ptr, 0, cursor), value)
            cursor += 1
        b.store(b.get_element_ptr(closure_ptr, 0, cursor), total_size)
        cursor += 1
        if reduction_ptr is not None:
            assert reduction is not None
            b.store(b.get_element_ptr(closure_ptr, 0, cursor), reduction_ptr)
            cursor += 1

        b.call(_KMPC_FORK_CALL, ident, IntValue(1, I32), inner_fn, closure_ptr)
        fork_tail = b

        # compile the microtask body, which runs on every thread of the region
        b = inner_fn.entry
        gtid = b.load(inner_fn.get_arg(0))
        inner_closure_ptr = inner_fn.get_arg(2)

        chunk = b.alloca(I32)
        lb = b.alloca(index_type)
        ub = b.alloca(index_type)
        step = b.alloca(index_type)
        local_sum_ptr = None
        if reduction is not None:
            assert reduction_llvm_type is not None
            local_sum_ptr = b.alloca(reduction_llvm_type)

        inner_args: list[Value] = []
        cursor = 0
        for _ in range(len(arg_lower_types)):
            inner_args.append(b.load(b.get_element_ptr(inner_closure_ptr, 0, cursor)))
            cursor += 1
        inner_total_size = b.load(b.get_element_ptr(inner_closure_ptr, 0, cursor))
        cursor += 1

        sum_ptr = None
        if reduction is not None:
            assert local_sum_ptr is not None
            sum_ptr = b.load(b.get_element_ptr(inner_closure_ptr, 0, cursor))
            reduction.store_initial_value(b, local_sum_ptr)
            cursor += 1

        b.store(chunk, 0)
        b.store(lb, 0)
        max_ub = b.sub(inner_total_size, IntValue(1, index_type))
        b.store(ub, max_ub)
        b.store(step, 1)

        b.call(
            kmpc_for_static_init,
            ident,
            gtid,
            IntValue(34, I32),
            chunk,
            lb,
            ub,
            step,
            IntValue(1, index_type),
            IntValue(1, index_type),
        )

        # clamp the upper bound returned by the runtime to [0, total_size - 1]
        clamper = BasicBlock()
        clamper.store(ub, max_ub)
        new_b = BasicBlock()
        clamper.jmp(new_b)
        b.br(b.icmp(IcmpOp.GT, True, b.load(ub), max_ub), clamper, new_b)
        b = new_b

        # main loop
        loop_builder = ForLoopBuilder(b, True, b.load(lb), b.load(ub), IntValue(1, index_type))
        b = loop_builder.body_entry
        b, value = kernel.compile_body(b, tuple(inner_args), loop_builder.loop_var, _DebugInterface(gtid))
        if reduction is not None:
            assert local_sum_ptr is not None
            b = reduction.reduce(b, local_sum_ptr, value)
        b = loop_builder.end(b)

        b.call(
            _KMPC_FOR_STATIC_FINI,
            ident,
            b.load(inner_fn.get_arg(0)),
        )

        if reduction is not None:
            assert local_sum_ptr is not None and sum_ptr is not None and reduction_llvm_type is not None
            sizeof_type = b.ptrtoint(b.get_element_ptr(NullValue(PointerType(reduction_llvm_type)), 1), _SIZE_T)

            reduce_fn = Function()
            reduce_fn.add_args(PointerType(reduction_llvm_type), PointerType(reduction_llvm_type))
            reduce_fn.set_return_type(VoidType())
            reduction.reduce(reduce_fn.entry, reduce_fn.get_arg(0), reduce_fn.entry.load(reduce_fn.get_arg(1)))
            reduce_fn.entry.ret(VoidValue())

            lock = GlobalZeroAggregateValue(_KMPC_CRITICAL_NAME)

            reduce_op = b.call(
                _KMPC_REDUCE_NOWAIT,
                ident,
                gtid,
                IntValue(1, I32),
                sizeof_type,
                sum_ptr,
                reduce_fn,
                lock,
            )
            op1_block = BasicBlock()
            op1_block = reduction.reduce(op1_block, sum_ptr, op1_block.load(local_sum_ptr))
            op1_block.call(
                _KMPC_END_REDUCE_NOWAIT,
                ident,
                gtid,
                lock,
            )
            op2_block = BasicBlock()
            op2_block = reduction.reduce(op2_block, sum_ptr, op2_block.load(local_sum_ptr), ordering=Ordering.MONOTONIC)

            b.br(b.icmp(IcmpOp.EQ, False, reduce_op, reduce_op.get_type().from_int(1)), op1_block, op2_block)
            new_block = BasicBlock()
            op1_block.jmp(new_block)
            op2_block.jmp(new_block)
            b = new_block

        b.ret(VoidValue())

        return fork_tail

def _echo_sync(b: BasicBlock, gtid: Value, *values: Value | str):
    gomp_critical = GlobalZeroAggregateValue(_KMPC_CRITICAL_NAME, flags=GlobalValueFlags.COMMON | GlobalValueFlags.GLOBAL)
    ident = GlobalAggregateValue(_IDEN_T,
        IntValue(0, I32),
        IntValue(0, I32),
        IntValue(0, I32),
        IntValue(0, I32),
        GlobalStringValue(b';unknown;unknown;0;0;;\00'),
    )
    b.call(_KMPC_CRITIAL, ident, gtid, gomp_critical)
    echo(b, *values)
    b.call(_KMPC_END_CRITICAL, ident, gtid, gomp_critical)
    _barrier(b, gtid)

def _barrier(b: BasicBlock, gtid: Value):
    ident = GlobalAggregateValue(_IDEN_T,
        IntValue(0, I32),
        IntValue(0, I32),
        IntValue(0, I32),
        IntValue(0, I32),
        GlobalStringValue(b';unknown;unknown;0;0;;\00'),
    )
    b.call(_KMPC_BARRIER, ident, gtid)
