from ctypes import CDLL
import ctypes
from os import environ

from typing import override
from llvmlite import binding as llvm

from .util import ForLoopBuilder, ctype_to_llvm
from .llvm import I32, BasicBlock, Function, GlobalAggregateValue, GlobalStringValue, IcmpOp, IntType, IntValue, Module, PointerType, StructType, Value, VoidType, VoidValue
from .backend import Backend, CompiledBackendFunction, LoopKernel

class ident_t(ctypes.Structure):
    _fields_ = [
        ('reserved1', ctypes.c_int32),
        ('reserved2', ctypes.c_int32),
        ('reserved3', ctypes.c_int32),
        ('reserved4', ctypes.c_int32),
        ('source', ctypes.POINTER(ctypes.c_int8)),
    ]

_LLVM_IDEN_T = ctype_to_llvm(ident_t)

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

_KMPC_FORK_CALL_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
)
_KMPC_FORK_CALL_CALLBACK._varargs_ = True  # type: ignore
_KMPC_FORK_CALL = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ident_t),
    ctypes.c_int32,
    _KMPC_FORK_CALL_CALLBACK,
)
_KMPC_FORK_CALL._varargs_ = True  # type: ignore

class ApiTable(ctypes.Structure):
    _fields_ = [
        ('kmpc_fork_call', _KMPC_FORK_CALL),
        ('kmpc_for_static_fini', ctypes.CFUNCTYPE(None, ctypes.POINTER(ident_t), ctypes.c_int)),
        ('kmpc_for_static_init_4', _for_static_init_type(ctypes.c_int32)),
        ('kmpc_for_static_init_8', _for_static_init_type(ctypes.c_int64)),
        ('kmpc_for_static_init_4u', _for_static_init_type(ctypes.c_uint32)),
        ('kmpc_for_static_init_8u', _for_static_init_type(ctypes.c_uint64)),
    ]

    _indices: dict[str, int] | None = None

    @staticmethod
    def get_fn_index(fn_name: str) -> int:
        if ApiTable._indices is None:
            ApiTable._indices = {name: index for index, (name, _) in enumerate(ApiTable._fields_)} # type: ignore
        return ApiTable._indices[fn_name]

    @staticmethod
    def get_for_static_init_fn_index(type: IntType) -> int:
        return ApiTable.get_fn_index('kmpc_for_static_init_' + str(type.bits // 8) + 'u')

    @staticmethod
    def from_cdll(cdll: CDLL) -> tuple['ApiTable', dict[str, type[ctypes._CDataType]]]:
        ret = ApiTable()
        objs: dict[str, type[ctypes._CDataType]] = {}
        for name, field_type in ApiTable._fields_: # type: ignore
            fn_ptr = ctypes.cast(getattr(cdll, '__' + name), field_type) # type: ignore
            setattr(ret, name, fn_ptr)
            objs[name] = fn_ptr
        return ret, objs

_LLVM_API_TABLE = ctype_to_llvm(ApiTable)

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
    _libomp: CDLL
    _api_table: ApiTable
    _api_objs: dict[str, type[ctypes._CDataType]]

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
        self._api_table, self._api_objs = ApiTable.from_cdll(self._libomp)

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
        closure_type = StructType(PointerType(_LLVM_API_TABLE), *arg_llvm_types, index_type)

        outer_fn = Function('main')
        outer_fn.add_args(PointerType(_LLVM_API_TABLE), *arg_llvm_types)
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
            kmpc_fork_call = b.load(b.get_element_ptr(api_table, 0, ApiTable.get_fn_index('kmpc_fork_call')))
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
            for i in range(len(arg_lower_types)):
                args.append(b.load(b.get_element_ptr(closure_ptr, 0, i + 1)))
            inner_total_size = b.load(b.get_element_ptr(closure_ptr, 0, len(arg_lower_types) + 1))

            b.store(chunk, 0)
            b.store(lb, 0)
            max_ub = b.sub(inner_total_size, IntValue(1, index_type))
            b.store(ub, max_ub)
            b.store(step, 1)
            kmpc_for_static_init = b.load(b.get_element_ptr(api_table, 0, ApiTable.get_for_static_init_fn_index(index_type)))
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

            kmpc_static_for_fini = b.load(b.get_element_ptr(api_table, 0, ApiTable.get_fn_index('kmpc_for_static_fini')))
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

        assert outer_fn.name is not None
        return _Compiled(self, mod, outer_fn.name, (ctypes.POINTER(ApiTable),) + tuple(a.to_ctype() for a in arg_lower_types))

class _Compiled(CompiledBackendFunction):
    _parent: OpenMPBackend
    _mod: Module
    _args_type: tuple[type[ctypes._CDataType], ...]
    _entry: ctypes._CFunctionType

    @override
    def __init__(self, parent: OpenMPBackend, mod: Module, entry_name: str, args_type: tuple[type[ctypes._CDataType], ...]) -> None:
        self._parent = parent
        self._mod = mod
        self._args_type = args_type

        target = llvm.Target.from_default_triple()
        tm = target.create_target_machine()

        llvm_mod = llvm.parse_assembly('\n'.join(mod.write()))
        llvm_mod.verify()

        engine = llvm.create_mcjit_compiler(llvm_mod, tm)
        engine.finalize_object()

        entry_type = ctypes.CFUNCTYPE(None, *self._args_type)
        self._entry = ctypes.cast(engine.get_function_address(entry_name), entry_type)

    @override
    def call(self, *args):
        return self._entry(ctypes.byref(self._parent._api_table), *args)

    @override
    def print_all(self) -> list[str]:
        return self._mod.write()
