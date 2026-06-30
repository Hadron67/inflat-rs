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
    _struct_type_cache: WeakKeyDictionary[StructType, type]
    def __init__(self) -> None:
        self._struct_type_cache = WeakKeyDictionary()

    def llvm_to_ctype(self, type: Type):
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
                s: Any = ctypes.Structure
                class _Ret(s):
                    _fields_ = tuple(self.llvm_to_ctype(a) for a in type.fields)
                self._struct_type_cache[type] = _Ret
                return _Ret
            case ArrayType():
                return self.llvm_to_ctype(type.child) * type.length
            case PointerType(child):
                fn: Any = ctypes.POINTER
                return fn(self.llvm_to_ctype(child))
            case FnType():
                return ctypes.CFUNCTYPE(self.llvm_to_ctype(type.return_type), *(self.llvm_to_ctype(a) for a in type.args))
            case _:
                raise TypeError(f"cannot convert to ctype: {type}")
