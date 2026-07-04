from inspect import isclass
from weakref import WeakKeyDictionary

import ctypes

from .llvm import I8, ArrayType, BasicBlock, FloatType, FnType, IcmpOp, IntType, Phi, PointerType, StructType, Type, Value, VoidType

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

    def llvm_may_void_to_ctype(self, type: Type) -> type[ctypes._CDataType] | None:
        if isinstance(type, VoidType):
            return None
        return self.llvm_to_ctype(type)

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
            case FloatType(bits):
                match bits:
                    case 32:
                        return ctypes.c_float
                    case 64:
                        return ctypes.c_double
                    case _:
                        raise ValueError(f"cannot convert to ctype: {type}")
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
                        return ctypes.CFUNCTYPE(self.llvm_may_void_to_ctype(child.return_type), *(self.llvm_to_ctype(a) for a in child.args))
                    else:
                        ret = ctypes.CFUNCTYPE(self.llvm_may_void_to_ctype(child.return_type))
                        ret.argtypes = None # pyright: ignore
                        ret.restype = None
                        return ret
                return ctypes.POINTER(self.llvm_to_ctype(child))
            case _:
                raise TypeError(f"cannot convert to ctype: {type}")

_llvm_struct_to_ctype: WeakKeyDictionary[StructType, type[ctypes._CDataType]] = WeakKeyDictionary()
def llvm_type_to_ctype(type: Type) -> type[ctypes._CDataType]:
    match type:
        case IntType(16):
            return ctypes.c_int16
        case IntType(32):
            return ctypes.c_int32
        case IntType(64):
            return ctypes.c_int64
        case FloatType(32):
            return ctypes.c_float
        case FloatType(64):
            return ctypes.c_double
        case VoidType():
            return ctypes.c_void_p
        case PointerType(child):
            return ctypes.POINTER(llvm_type_to_ctype(child))
        case StructType():
            if type in _llvm_struct_to_ctype:
                return _llvm_struct_to_ctype[type]
            class _Struct(ctypes.Structure):
                _fields_ = []
            _llvm_struct_to_ctype[type] = _Struct
            for field in type.fields:
                _Struct._fields_.append((f"f{len(_Struct._fields_)}", llvm_type_to_ctype(field))) # type: ignore
            return _Struct
    raise TypeError(f"cannot convert to ctype: {type}")

_ctype_struct_to_llvm: WeakKeyDictionary[type[ctypes.Structure], StructType] = WeakKeyDictionary()
def ctype_to_llvm(type: type[ctypes._CDataType]) -> Type:
    if issubclass(type, ctypes.Structure):
        if type in _ctype_struct_to_llvm:
            return _ctype_struct_to_llvm[type]
        ret = StructType()
        _ctype_struct_to_llvm[type] = ret
        for field in type._fields_:
            ret.add_field(ctype_to_llvm(field[1]))
        return ret
    if issubclass(type, ctypes._Pointer):
        return PointerType(ctype_to_llvm(type._type_)) # pyright: ignore
    if issubclass(type, ctypes._CFuncPtr): # type: ignore
        vararg = getattr(type, '_varargs_', False)
        restype = None
        if type._restype_ is None: # type: ignore
            restype = VoidType()
        else:
            restype = ctype_to_llvm(type._restype_) # type: ignore
        return PointerType(FnType(tuple(ctype_to_llvm(a) for a in type._argtypes_), restype, vararg)) #type: ignore
    match type:
        case ctypes.c_int8 | ctypes.c_uint8:
            return IntType(8)
        case ctypes.c_int16 | ctypes.c_uint16:
            return IntType(16)
        case ctypes.c_int32 | ctypes.c_uint32:
            return IntType(32)
        case ctypes.c_int64 | ctypes.c_uint64:
            return IntType(64)
        case ctypes.c_float:
            return FloatType(32)
        case ctypes.c_double:
            return FloatType(64)
        case _:
            raise TypeError(f"cannot convert to llvm type: {type}")
