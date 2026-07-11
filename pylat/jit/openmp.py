from ctypes import CDLL
import ctypes

from typing import override
from llvmlite import binding as llvm

from .helper import echo, _GLOBAL_HELPERS

from .util import ForLoopBuilder
from .llvm import I32, I64, I8, ArrayType, BasicBlock, DeclareFunction, FnType, Function, GlobalAggregateValue, GlobalStringValue, GlobalValueFlags, GlobalZeroAggregateValue, IcmpOp, IntType, IntValue, Module, NullValue, Ordering, PointerType, StructType, Value, VoidType, VoidValue, fn_type
from .backend import Backend, CompiledBackendFunction, DebugInterface, LoopKernel, ReductionKernel

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
    def compile_paralell_loop(self, kernel: LoopKernel, reduction: ReductionKernel | None = None) -> CompiledBackendFunction:
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
        if reduction is not None:
            closure_type.add_field(PointerType(PointerType(reduction.get_type())))

        outer_fn = Function('main')
        outer_fn.add_args(*arg_llvm_types)
        outer_fn.set_return_type(VoidType())

        inner_fn = Function()
        inner_fn.add_args(PointerType(I32), PointerType(I32))
        inner_fn.set_return_type(VoidType(), True)
        inner_fn.add_args(PointerType(closure_type))

        kmpc_for_static_init = _for_static_init(index_type)

        # separate outer and inner function subroutines so that we won't accidentally use a variable across functions
        def compile_outer():
            b = outer_fn.entry
            closure_ptr = b.alloca(closure_type)

            reduction_ptr = None
            if reduction is not None:
                reduction_ptr = b.alloca(PointerType(reduction.get_type()))
                reduction.store_initial_value(b, reduction_ptr)

            args = outer_fn.get_args()
            b, total_size = kernel.compile_total_size(b, args)
            cursor = 0
            for value in args:
                b.store(b.get_element_ptr(closure_ptr, 0, cursor), value)
                cursor += 1
            b.store(b.get_element_ptr(closure_ptr, 0, cursor), total_size)
            cursor += 1
            if reduction_ptr is not None:
                b.store(b.get_element_ptr(closure_ptr, 0, cursor), reduction_ptr)
                cursor += 1

            b.call(_KMPC_FORK_CALL, ident, IntValue(1, I32), inner_fn, closure_ptr)
            b.ret(VoidValue())

        def compile_inner():
            b = inner_fn.entry
            gtid = b.load(inner_fn.get_arg(0))
            closure_ptr = inner_fn.get_arg(2)

            chunk = b.alloca(I32)
            lb = b.alloca(index_type)
            ub = b.alloca(index_type)
            step = b.alloca(index_type)
            local_sum_ptr = None
            reduce_data = None
            if reduction is not None:
                local_sum_ptr = b.alloca(reduction.get_type())
                reduce_data = b.alloca(PointerType(reduction.get_type()))

            args: list[Value] = []
            cursor = 0
            for _ in range(len(arg_lower_types)):
                args.append(b.load(b.get_element_ptr(closure_ptr, 0, cursor)))
                cursor += 1
            inner_total_size = b.load(b.get_element_ptr(closure_ptr, 0, cursor))
            cursor += 1

            sum_ptr = None
            if reduction is not None:
                assert local_sum_ptr is not None
                sum_ptr = b.load(b.get_element_ptr(closure_ptr, 0, cursor))
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

            # clamp upper bound
            clamper = BasicBlock()
            clamper.store(ub, max_ub)
            new_b = BasicBlock()
            clamper.jmp(new_b)
            b.br(b.icmp(IcmpOp.GT, True, b.load(ub), max_ub), clamper, new_b)
            b = new_b

            # main loop
            loop_builder = ForLoopBuilder(b, True, b.load(lb), b.load(ub), IntValue(1, index_type))
            b = loop_builder.body_entry
            b, value = kernel.compile_body(b, tuple(args), loop_builder.loop_var, _DebugInterface(gtid))
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
                assert reduce_data is not None and local_sum_ptr is not None and sum_ptr is not None
                reduce_type = reduction.get_type()
                sizeof_type = b.ptrtoint(b.get_element_ptr(NullValue(PointerType(reduce_type)), 1), _SIZE_T)

                reduce_fn = Function()
                reduce_fn.add_args(PointerType(reduce_type), PointerType(reduce_type))
                reduce_fn.set_return_type(VoidType())
                reduction.reduce(reduce_fn.entry, reduce_fn.get_arg(0), reduce_fn.entry.load(reduce_fn.get_arg(1)))

                lock = GlobalZeroAggregateValue(_KMPC_CRITICAL_NAME)

                b.store(reduce_data, local_sum_ptr)

                reduce_op = b.call(
                    _KMPC_REDUCE_NOWAIT,
                    ident,
                    gtid,
                    sizeof_type,
                    reduce_data,
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

        compile_outer()
        compile_inner()

        mod = Module()
        mod.add_recursively(values=[outer_fn])

        assert outer_fn.name is not None

        return _Compiled(self, mod, outer_fn.name, tuple(a.to_ctype() for a in arg_lower_types))

class _Compiled(CompiledBackendFunction):
    _parent: OpenMPBackend
    _mod: Module
    _args_type: tuple[type[ctypes._CDataType], ...]
    _entry: ctypes._CFunctionType
    _engine: llvm.ExecutionEngine

    @override
    def __init__(self, parent: OpenMPBackend, mod: Module, entry_name: str, args_type: tuple[type[ctypes._CDataType], ...]) -> None:
        self._parent = parent
        self._mod = mod
        self._args_type = args_type

        target = llvm.Target.from_default_triple()
        tm = target.create_target_machine()

        llvm_mod = llvm.parse_assembly('\n'.join(mod.write()))
        llvm_mod.verify()

        backing_mod = llvm.parse_assembly("")
        engine = llvm.create_mcjit_compiler(backing_mod, tm)
        self._engine = engine
        engine.add_module(llvm_mod)

        libc = ctypes.CDLL(None)
        for name in [_KMPC_FORK_CALL.name, _KMPC_FOR_STATIC_FINI.name, _KMPC_CRITIAL.name, _KMPC_END_CRITICAL.name, '__kmpc_for_static_init_8']:
            value = None
            try:
                value = llvm_mod.get_function(name)
            except NameError:
                continue
            engine.add_global_mapping(
                value,
                ctypes.cast(getattr(libc, name), ctypes.c_void_p).value,
            )
        for name, addr in _GLOBAL_HELPERS:
            try:
                engine.add_global_mapping(
                    llvm_mod.get_function(name),
                    addr,
                )
            except NameError:
                pass
        engine.finalize_object()
        engine.run_static_constructors()

        entry_type = ctypes.CFUNCTYPE(None, *self._args_type)
        addr = engine.get_function_address(entry_name)
        self._entry = ctypes.cast(addr, entry_type)

    @override
    def call(self, *args):
        return self._entry(*args)

    @override
    def print_all(self) -> list[str]:
        return self._mod.write()

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
