from typing import Any
from weakref import WeakKeyDictionary

from llvmlite.binding.orcjit import ctypes

from .llvm import ArrayType, BasicBlock, Float32Type, Float64Type, FnType, IcmpOp, IntType, Phi, PointerType, StructType, Type, Value

class ForLoopBuilder:
    _entry: BasicBlock
    _step: Value
    _phi: Phi
    _comp: BasicBlock
    _body_entry: BasicBlock
    _output_block: BasicBlock

    def __init__(self, entry: BasicBlock, signed: bool, lower: Value, upper: Value, step: Value) -> None:
        self._entry = entry
        self._comp = BasicBlock()
        self._body_entry = BasicBlock()
        self._output_block = BasicBlock()
        self._step = step

        entry.jmp(self._comp)
        self._phi = self._comp.phi((lower, entry))
        self._comp.br(
            self._comp.icmp(IcmpOp.LE, signed, self._phi, upper),
            self._body_entry,
            self._output_block,
        )

    def end(self, body_end: BasicBlock):
        next_var = body_end.add(self._phi, self._step)
        body_end.jmp(self._comp)
        self._phi.add_incoming(next_var, body_end)
        return self._output_block

    @property
    def loop_var(self):
        return self._phi

    @property
    def body_entry(self):
        return self._body_entry

class TypeConverter:
    _struct_type_cache: WeakKeyDictionary[StructType, type[ctypes.Structure]]
    def __init__(self) -> None:
        self._struct_type_cache = WeakKeyDictionary()

    def llvm_to_ctype(self, type: Type) -> type[ctypes._CDataType]:
        match type:
            case IntType(bits):
                match bits:
                    case 8:
                        return ctypes.c_int8
                    case 16:
                        return ctypes.c_int16
                    case 32:
                        return ctypes.c_int32
                    case 64:
                        return ctypes.c_int64
                    case _:
                        raise TypeError(f"cannot convert to ctype: {type}")
            case Float32Type():
                return ctypes.c_float
            case Float64Type():
                return ctypes.c_double
            case StructType():
                if type in self._struct_type_cache:
                    return self._struct_type_cache[type]
                class _Ret(ctypes.Structure):
                    _fields_ = tuple((f'field{i}', self.llvm_to_ctype(a)) for i, a in enumerate(type.fields))
                self._struct_type_cache[type] = _Ret
                return _Ret
            case ArrayType():
                return self.llvm_to_ctype(type.child) * type.length
            case PointerType(child):
                if isinstance(child, FnType):
                    if not child.varargs:
                        return ctypes.CFUNCTYPE(self.llvm_to_ctype(child.return_type), *(self.llvm_to_ctype(a) for a in child.args))
                    else:
                        ret = ctypes.CFUNCTYPE(self.llvm_to_ctype(child.return_type))
                        ret.argtypes = None # pyright: ignore
                        return ret
                return ctypes.POINTER(self.llvm_to_ctype(child))
            case _:
                raise TypeError(f"cannot convert to ctype: {type}")
