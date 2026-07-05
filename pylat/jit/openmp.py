from ctypes import CDLL
import ctypes

from typing import override
from llvmlite import binding as llvm

from .helper import echo, _GLOBAL_HELPERS

from .util import ForLoopBuilder
from .llvm import I32, I8, ArrayType, BasicBlock, DeclareFunction, FnType, Function, GlobalAggregateValue, GlobalStringValue, GlobalValueFlags, GlobalZeroAggregateValue, IcmpOp, IntType, IntValue, Module, PointerType, StructType, Value, VoidType, VoidValue
from .backend import Backend, CompiledBackendFunction, DebugInterface, LoopKernel

class ident_t(ctypes.Structure):
    _fields_ = [
        ('reserved1', ctypes.c_int32),
        ('reserved2', ctypes.c_int32),
        ('reserved3', ctypes.c_int32),
        ('reserved4', ctypes.c_int32),
        ('source', ctypes.POINTER(ctypes.c_int8)),
    ]

_LLVM_IDEN_T = StructType(
    I32,
    I32,
    I32,
    I32,
    PointerType(I8),
)

def _for_static_init_type(type: type[ctypes._SimpleCData]):
    return ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ident_t), # loc
        ctypes.c_uint32, # gitd
        ctypes.c_uint32, # schedtype
        ctypes.POINTER(ctypes.c_int32), # plastiter
        ctypes.POINTER(type), # plower
        ctypes.POINTER(type), # pupper
        ctypes.POINTER(type), # pstride
        type, # incr
        type, # chunk
    )

def _for_static_init(type: IntType):
    fn_type = FnType((
        PointerType(_LLVM_IDEN_T), # loc
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

_KMPC_FORK_CALL_CALLBACK_TYPE = FnType((
    PointerType(I32),
    PointerType(I32),
), VoidType(), True)
_KMPC_FORK_CALL_TYPE = FnType((
    PointerType(_LLVM_IDEN_T),
    I32,
    PointerType(_KMPC_FORK_CALL_CALLBACK_TYPE),
), VoidType(), True)

_KMPC_FORK_CALL = DeclareFunction('__kmpc_fork_call', _KMPC_FORK_CALL_TYPE)
_KMPC_FOR_STATIC_FINI = DeclareFunction('__kmpc_for_static_fini', FnType((PointerType(_LLVM_IDEN_T), I32), VoidType()))
_KMPC_CRITIAL = DeclareFunction('__kmpc_critical', FnType((PointerType(_LLVM_IDEN_T), I32, PointerType(ArrayType(I32, 8))), VoidType()))
_KMPC_END_CRITICAL = DeclareFunction('__kmpc_end_critical', FnType((PointerType(_LLVM_IDEN_T), I32, PointerType(ArrayType(I32, 8))), VoidType()))
_KMPC_BARRIER = DeclareFunction('__kmpc_barrier', FnType((PointerType(_LLVM_IDEN_T), I32), VoidType()))

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
    def compile_paralell_loop(self, kernel: LoopKernel) -> CompiledBackendFunction:
        index_type = kernel.get_index_type()
        arg_lower_types = kernel.get_args()
        arg_llvm_types = tuple(a.to_llvm_type() for a in arg_lower_types)
        ident = GlobalAggregateValue(_LLVM_IDEN_T,
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            GlobalStringValue(b';unknown;unknown;0;0;;\00'),
        )
        closure_type = StructType(*arg_llvm_types, index_type)

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
            echo(b, "start")
            closure_ptr = b.alloca(closure_type)
            args = outer_fn.get_args()
            b, total_size = kernel.compile_total_size(b, args)
            for i, value in enumerate(args + (total_size,)):
                b.store(b.get_element_ptr(closure_ptr, 0, i), value)
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

            args: list[Value] = []
            for i in range(len(arg_lower_types)):
                args.append(b.load(b.get_element_ptr(closure_ptr, 0, i)))
            inner_total_size = b.load(b.get_element_ptr(closure_ptr, 0, len(arg_lower_types)))

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

            loop_builder = ForLoopBuilder(b, True, b.load(lb), b.load(ub), IntValue(1, index_type))
            b = loop_builder.body_entry
            b = kernel.compile_body(b, tuple(args), loop_builder.loop_var, _DebugInterface(gtid))
            b = loop_builder.end(b)

            b.call(
                _KMPC_FOR_STATIC_FINI,
                ident,
                b.load(inner_fn.get_arg(0)),
            )
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
    gomp_critical = GlobalZeroAggregateValue(ArrayType(I32, 8), flags=GlobalValueFlags.COMMON | GlobalValueFlags.GLOBAL)
    ident = GlobalAggregateValue(_LLVM_IDEN_T,
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
    ident = GlobalAggregateValue(_LLVM_IDEN_T,
        IntValue(0, I32),
        IntValue(0, I32),
        IntValue(0, I32),
        IntValue(0, I32),
        GlobalStringValue(b';unknown;unknown;0;0;;\00'),
    )
    b.call(_KMPC_BARRIER, ident, gtid)
