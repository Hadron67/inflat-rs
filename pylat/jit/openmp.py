from ctypes import CDLL
import ctypes
from os import environ

from typing_extensions import override

from pylat.jit.util import ForLoopBuilder

from .llvm import I32, I64, I8, AggregateValue, BasicBlock, FnType, Function, GlobalDefineValue, IcmpOp, IntType, IntValue, Module, PointerType, StringLiteralValue, StructType, Type, Value, VoidType, VoidValue
from .backend import Backend, CompiledBackendFunction, ParallelForLoopProvider

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
    def get_for_loop_provider(self) -> ParallelForLoopProvider:
        return OpenMPParallelForProvider(self)

class OpenMPParallelForProvider(ParallelForLoopProvider):
    _parent: OpenMPBackend
    _api_table_type: StructType

    _outer_fn: Function
    _outer_closure_ptr: Value | None
    _outer_api_table: Value
    _in_loop: bool

    _inner_fn: Function
    _inner_api_table: Value
    _outer_closure_vars: list[Value]
    _outer_args_to_closure_map: dict[int, Value]
    _closure_type: StructType
    _inner_unpack_closure_block: BasicBlock
    _inner_loop_builder: ForLoopBuilder | None
    _ident: GlobalDefineValue

    _prologue_end: BasicBlock | None

    @override
    def __init__(self, parent: OpenMPBackend) -> None:
        self._parent = parent
        self._api_table_type = StructType(PointerType(_Types.KMPC_FORK_CALL), PointerType(_Types.KMPC_FOR_STATIC_FINI))
        self._ident = GlobalDefineValue(AggregateValue(_Types.IDENT_T,
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            IntValue(0, I32),
            GlobalDefineValue(StringLiteralValue(b';unknown;unknown;0;0;;\00')),
        ))

        self._outer_closure_vars = []
        self._outer_args_to_closure_map = {}
        self._closure_type = StructType()
        self._inner_loop_builder = None
        self._prologue_end = None
        self._outer_closure_ptr = None

        self._in_loop = False
        self._outer_fn = Function("__main")
        outer_api_table, = self._outer_fn.add_args(PointerType(self._api_table_type))
        self._outer_api_table = outer_api_table

        self._inner_fn = Function()
        self._inner_fn.add_args(PointerType(I32), PointerType(I32))
        self._inner_fn.set_return_type(VoidType(), True)
        self._inner_fn.add_arg(PointerType(self._closure_type))
        self._inner_unpack_closure_block = BasicBlock()

        self._inner_api_table = self._tunnel_through_closure(outer_api_table)

    def _tunnel_through_closure(self, value: Value):
        index = len(self._outer_closure_vars)
        self._outer_closure_vars.append(value)
        self._closure_type.add_field(value.get_type())
        return self._inner_unpack_closure_block.load(
            self._inner_unpack_closure_block.get_element_ptr(
                self._inner_fn.get_arg(2),
                IntValue(0, I64),
                IntValue(index, I32),
            )
        )

    @override
    def begin_prologue(self) -> BasicBlock:
        b = self._outer_fn.entry
        self._outer_closure_ptr = b.alloca(self._closure_type)
        return b

    @override
    def add_arg(self, type: Type) -> int:
        return self._outer_fn.add_arg(type)

    @override
    def get_arg(self, index: int) -> Value:
        if not self._in_loop:
            return self._outer_fn.get_arg(index)

        if index in self._outer_args_to_closure_map:
            return self._outer_args_to_closure_map[index]

        ret = self._tunnel_through_closure(self._outer_fn.get_arg(index))
        self._outer_args_to_closure_map[index] = ret
        return ret

    @override
    def begin_loop(self, index_type: IntType, prologue_end: BasicBlock, total_size: Value) -> tuple[BasicBlock, Value]:
        self._prologue_end = prologue_end

        init_fn_name, init_fn_type = _create_kmpc_for_static_init(index_type, True)
        self._api_table_type.add_field(PointerType(init_fn_type))

        total_size = self._tunnel_through_closure(total_size)
        b = self._inner_fn.entry

        chunk = b.alloca(I32)
        lb = b.alloca(index_type)
        ub = b.alloca(index_type)
        step = b.alloca(index_type)

        b.jmp(self._inner_unpack_closure_block)
        b = self._inner_unpack_closure_block

        b.store(chunk, 0)
        b.store(lb, 0)
        max_ub = b.sub(total_size, IntValue(1, index_type))
        b.store(ub, max_ub)
        b.store(step, 1)
        kmpc_for_static_init = b.load(b.get_element_ptr(self._inner_api_table, IntValue(0, I64), IntValue(2, I32)))
        b.call(
            kmpc_for_static_init,
            self._ident,
            b.load(self._inner_fn.get_arg(0)),
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

        self._inner_loop_builder = ForLoopBuilder(b, True, b.load(lb), b.load(ub), IntValue(1, index_type))

        self._in_loop = True
        return self._inner_loop_builder.body_entry, self._inner_loop_builder.loop_var

    def end(self, block: BasicBlock) -> CompiledBackendFunction:
        assert self._inner_loop_builder is not None
        b = self._inner_loop_builder.end(block)
        self._inner_loop_builder = None
        kmpc_static_for_fini = b.load(b.get_element_ptr(self._inner_api_table, IntValue(0, I64), IntValue(1, I32)))
        b.call(
            kmpc_static_for_fini,
            self._ident,
            b.load(self._inner_fn.get_arg(0)),
        )
        b.ret(VoidValue())

        assert self._prologue_end is not None
        if self._prologue_end is not self._outer_fn.entry:
            self._outer_fn.entry.jmp(self._prologue_end)
        b = self._prologue_end

        assert self._outer_closure_ptr is not None
        for i, value in enumerate(self._outer_closure_vars):
            b.store(b.get_element_ptr(self._outer_closure_ptr, IntValue(0, I64), IntValue(i, I32)), value)

        kmpc_fork_call = b.load(b.get_element_ptr(self._outer_api_table, IntValue(0, I64), IntValue(0, I32)))
        b.call(kmpc_fork_call, self._ident, IntValue(0, I32), self._inner_fn, self._outer_closure_ptr)
        b.ret(VoidValue())

        self._outer_fn.set_return_type(VoidType())

        mod = Module()
        mod.add_recursively(values=[self._outer_fn])
        return _Compiled(self._parent, mod)

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
