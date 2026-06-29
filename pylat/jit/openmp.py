from ctypes import CDLL
import ctypes
from os import environ

from typing_extensions import override

from pylat.jit.util import ForLoopBuilder

from .llvm import I32, I64, I8, AggregateValue, BasicBlock, FnType, Function, GlobalDefineValue, IcmpOp, IntType, IntValue, Module, PointerType, StringLiteralValue, StructType, Type, Value, VoidType, VoidValue
from .backend import Backend, CompiledBackendFunction, LoopKernel, ParallelForLoopProvider

class _Types:
    IDENT_T = StructType(I32, I32, I32, I32, PointerType(I8))
    KMPC_FORK_CALL_CALLBACK = FnType((PointerType(I32), PointerType(I32)), VoidType(), True)
    KMPC_FORK_CALL = FnType((
        PointerType(IDENT_T),
        I32,
        PointerType(KMPC_FORK_CALL_CALLBACK),
    ), VoidType(), True)
    KMPC_FOR_STATIC_FINI = FnType((PointerType(IDENT_T), I32), VoidType())

def _create_kmpc_for_static_init(type: IntType, signed: bool):
    name = f"__kmpc_for_static_init_{type.bits // 8}{'u' if not signed else ''}"
    fn_type = FnType((
        PointerType(_Types.IDENT_T), # loc
        I32, # gtid
        I32, # schedtype
        PointerType(I32), # plastiter
        PointerType(type), # plower
        PointerType(type), # pupper
        PointerType(type), # pstride
        type, # incr
        type, # chunk
    ), VoidType())
    return name, fn_type

def _try_open_libomp(try_paths: list[str]) -> CDLL:
    path = environ.get("PYLAT_LIBOMP", None)
    if path is not None:
        return ctypes.cdll.LoadLibrary(path)
    for p in try_paths:
        try:
            return ctypes.cdll.LoadLibrary(p)
        except OSError:
            pass
    raise RuntimeError(f"failed to load libomp, tries paths: {', '.join(try_paths)}")

class OpenMPBackend(Backend):
    def __init__(self, libomp: str | CDLL | None = None) -> None:
        match libomp:
            case str():
                self._libomp = ctypes.cdll.LoadLibrary(libomp)
            case CDLL():
                self._libomp = libomp
            case _:
                self._libomp = _try_open_libomp([
                    "/usr/lib/libomp.so",
                    "/opt/homebrew/opt/llvm/lib/libomp.dylib",
                ])

    @override
    def compile_paralell_loop(self, kernel: LoopKernel) -> CompiledBackendFunction:
        index_type = kernel.get_index_type()
        arg_types = kernel.get_args()
        ident = GlobalDefineValue(AggregateValue(_Types.IDENT_T,
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            GlobalDefineValue(StringLiteralValue(b';unknown;unknown;0;0;;\00')),
        ))
        init_fn_name, init_fn_type = _create_kmpc_for_static_init(index_type, True)
        api_table_type = StructType(
            PointerType(_Types.KMPC_FORK_CALL),
            PointerType(_Types.KMPC_FOR_STATIC_FINI),
            PointerType(init_fn_type),
        )
        closure_type = StructType(PointerType(api_table_type), *arg_types, index_type)

        outer_fn = Function('main')
        outer_fn.add_args(PointerType(api_table_type), *arg_types)
        outer_fn.set_return_type(VoidType())

        inner_fn = Function()
        inner_fn.add_args(PointerType(I32), PointerType(I32))
        inner_fn.set_return_type(VoidType(), True)
        inner_fn.add_arg(PointerType(closure_type))

        # separate outer and inner function subroutines so that we won't accidentally use a variable across functions
        def compile_outer():
            b = outer_fn.entry
            closure_ptr = b.alloca(closure_type)
            args = outer_fn.get_args()[1:]
            b, total_size = kernel.compile_total_size(b, args)
            for i, value in enumerate(outer_fn.get_args() + (total_size,)):
                b.store(b.get_element_ptr(closure_ptr, 0, i), value)
            api_table = outer_fn.get_arg(0)
            kmpc_fork_call = b.load(b.get_element_ptr(api_table, 0, 0))
            b.call(kmpc_fork_call, ident, IntValue(0, I32), inner_fn, closure_ptr)
            b.ret(VoidValue())

        def compile_inner():
            b = inner_fn.entry
            closure_ptr = inner_fn.get_arg(2)

            chunk = b.alloca(I32)
            lb = b.alloca(index_type)
            ub = b.alloca(index_type)
            step = b.alloca(index_type)

            args: list[Value] = []
            api_table = b.load(b.get_element_ptr(closure_ptr, 0, 0))
            for i in range(len(arg_types)):
                args.append(b.load(b.get_element_ptr(closure_ptr, 0, i + 1)))
            inner_total_size = b.load(b.get_element_ptr(closure_ptr, 0, len(arg_types) + 1))

            b.store(chunk, 0)
            b.store(lb, 0)
            max_ub = b.sub(inner_total_size, IntValue(1, index_type))
            b.store(ub, max_ub)
            b.store(step, 1)
            kmpc_for_static_init = b.load(b.get_element_ptr(api_table, 0, 2))
            b.call(
                kmpc_for_static_init,
                ident,
                b.load(inner_fn.get_arg(0)),
                IntValue(43, I32),
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
            b = kernel.compile_body(loop_builder.body_entry, tuple(args), loop_builder.loop_var)
            b = loop_builder.end(b)

            kmpc_static_for_fini = b.load(b.get_element_ptr(api_table, IntValue(0, I64), IntValue(1, I32)))
            b.call(
                kmpc_static_for_fini,
                ident,
                b.load(inner_fn.get_arg(0)),
            )
            b.ret(VoidValue())

        compile_outer()
        compile_inner()

        mod = Module()
        mod.add_recursively(values=[outer_fn])

        return _Compiled(self, mod)

class _Compiled(CompiledBackendFunction):
    _parent: OpenMPBackend
    _mod: Module

    @override
    def __init__(self, parent: OpenMPBackend, mod: Module) -> None:
        self._parent = parent
        self._mod = mod

    @override
    def call(self, *args):
        raise NotImplementedError

    @override
    def print_all(self) -> list[str]:
        return self._mod.write()
