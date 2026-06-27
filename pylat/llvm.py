from enum import IntEnum
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import override

from .util import next_unique_name

class Type:
    def from_float(self, value: float) -> 'Value':
        raise TypeError(f'cannot create float from {self}')

    def from_int(self, value: int) -> 'Value':
        return self.from_float(float(value))

class FloatType(Type):
    @abstractmethod
    def bits(self) -> int:
        pass

@dataclass
class HalfFloatType(FloatType):
    @override
    def from_float(self, value: float) -> 'Value':
        # TODO: check range
        return FloatValue(value, self)

    def __str__(self) -> str:
        return 'half'

    @override
    def bits(self) -> int:
        return 16

@dataclass
class Float32Type(FloatType):
    def __str__(self) -> str:
        return 'float'

    @override
    def bits(self) -> int:
        return 32

    @override
    def from_float(self, value: float) -> 'Value':
        # TODO: check range
        return FloatValue(value, self)

@dataclass
class Float64Type(FloatType):
    def __str__(self) -> str:
        return 'double'

    @override
    def bits(self) -> int:
        return 64

    @override
    def from_float(self, value: float) -> 'Value':
        # TODO: check range
        return FloatValue(value, self)

@dataclass
class IntType(Type):
    bits: int
    signed: bool

    def __str__(self) -> str:
        return f'{"i"}{self.bits}'

    def get_range(self) -> tuple[int, int]:
        if self.signed:
            return (-2**(self.bits - 1), 2**(self.bits - 1) - 1)
        else:
            return (0, 2**self.bits - 1)

    @override
    def from_int(self, value: int) -> 'Value':
        min, max = self.get_range()
        if not min <= value <= max:
            raise ValueError(f'value {value} out of range for {self}')
        return IntValue(value, self)

@dataclass
class LabelType(Type):
    def __str__(self) -> str:
        return 'label'

@dataclass
class PointerType(Type):
    child: Type

    def __str__(self) -> str:
        return 'ptr'

@dataclass
class FnType(Type):
    args: tuple[Type, ...]
    return_type: Type
    varargs: bool = False

    def __str__(self) -> str:
        return f'fn({",".join(str(arg) for arg in self.args)}) -> {self.return_type}'

@dataclass
class VoidType(Type):
    def __str__(self) -> str:
        return 'void'

@dataclass
class StructType(Type):
    name: str
    fields: tuple[Type, ...]

    def __str__(self) -> str:
        return '%' + self.name

    def write_definition(self) -> list[str]:
        return [f"%{self.name} = type {{{', '.join(str(i) for i in self.fields)}}}"]

class Value:
    @abstractmethod
    def get_type(self) -> Type:
        pass

@dataclass
class RegisterValue(Value):
    name: str
    type: Type

    def __str__(self) -> str:
        return str(self.type) + ' %' + self.name

    @override
    def get_type(self) -> Type:
        return self.type

@dataclass
class ArgValue(RegisterValue):
    index: int

@dataclass
class VoidValue(Value):
    def __str__(self) -> str:
        return 'void'

    @override
    def get_type(self) -> Type:
        return VoidType()

@dataclass
class IntValue(Value):
    value: int
    type: IntType

    def __str__(self) -> str:
        return str(self.type) + ' ' + str(self.value)

    @override
    def get_type(self) -> Type:
        return self.type

@dataclass
class FloatValue(Value):
    value: float
    type: FloatType

    def __str__(self) -> str:
        return str(self.value)

    @override
    def get_type(self) -> Type:
        return self.type

class Module:
    globals: 'dict[str, GlobalValue]'
    struct_types: 'dict[str, StructType]'

    def __init__(self) -> None:
        self.globals = {}
        self.struct_types = {}

    def next_unique_global_name(self, prefix: str='') -> str:
        i = 0
        while True:
            name = prefix + str(i)
            if name not in self.globals:
                return name
            i += 1

    def next_unique_type_name(self, prefix: str = '') -> str:
        i = 0
        while True:
            name = prefix + str(i)
            if name not in self.struct_types:
                return name
            i += 1

    def add_global(self, value: 'GlobalValue'):
        if value.name not in self.globals:
            self.globals[value.name] = value

    def add_struct_type(self, name_prefix: str, fields: tuple[Type, ...]) -> StructType:
        ret = StructType(self.next_unique_type_name(name_prefix), fields)
        self.struct_types[ret.name] = ret
        return ret

    def add_function(self, name_prefix: str = 'fn.') -> 'Function':
        ret = Function(self.next_unique_global_name(name_prefix))
        self.globals[ret.name] = ret
        return ret

    def write(self) -> list[str]:
        ret: list[str] = []
        for type in self.struct_types.values():
            ret.extend(type.write_definition())
        for value in self.globals.values():
            ret.extend(value.write_definition())
        return ret

@dataclass
class GlobalValue(Value):
    name: str
    internal: bool

    def __str__(self) -> str:
        return '@' + self.name

    @abstractmethod
    def write_definition(self) -> list[str]:
        pass

@dataclass
class GlobalDefineValue(GlobalValue):
    value: Value
    is_const: bool = True
    is_private: bool = True
    is_unnamed_addr: bool = True

    @override
    def write_definition(self) -> list[str]:
        flags: list[str] = []
        if self.is_private:
            flags.append('private')
        if self.is_unnamed_addr:
            flags.append('unnamed_addr')
        if self.is_const:
            flags.append('constant')
        return [f'@{self.name} = {' '.join(flags)}{self.value}']

class DeclareFunction(GlobalValue):
    type: FnType

    def __init__(self, name: str, type: FnType, internal: bool = False):
        self.name = name
        self.type = type

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def write_definition(self) -> list[str]:
        args: list[str] = []
        for arg in self.type.args:
            args.append(str(arg))
        if self.type.varargs:
            args.append('...')
        return [f"declare {self.type.return_type} @{self.name}({', '.join(args)})"]

class FunctionState(IntEnum):
    BUILDING_ARGS = 0
    BUILDING_RETURN_TYPE = 1
    BUILDING_BODY = 2
    FINISHED = 3

class Function(GlobalValue):
    blocks: 'list[BasicBlock]'
    _args: list[RegisterValue]

    _used_names: set[str] = set()
    _type: FnType | None

    def __init__(self, name: str, internal: bool = False) -> None:
        self.name = name
        self.blocks = []
        self._args = []
        self.type = None

        self._used_names = set()

        self.internal = internal

    def __str__(self) -> str:
        return '@' + self.name

    @override
    def write_definition(self) -> list[str]:
        assert self.type is not None
        ret: list[str] = []
        ret.append(f'define {'internal ' if self.internal else ''}{self.type.return_type} @{self.name}({', '.join(str(i) for i in self._args)}) {{')
        for block in self.blocks:
            ret.append(block.name + ':')
            for line in block.write():
                ret.append('    ' + line)
        ret.append('}')
        return ret

    def next_unique_name(self, prefix: str='') -> str:
        return next_unique_name(prefix, self._used_names)

    @override
    def get_type(self) -> Type:
        assert self.type is not None
        return self.type

    def add_arg(self, type: Type, name_prefix: str | None = None) -> ArgValue:
        assert self._type is None
        ret = ArgValue(self.next_unique_name(name_prefix if name_prefix is not None else 'arg.'), type, len(self._args))
        self._args.append(ret)
        return ret

    def set_return_type(self, type: Type):
        assert self._type is None
        self._type = FnType(tuple(i.type for i in self._args), type)

    def add_block(self, name_prefix: str | None = None) -> 'BasicBlockBuilder':
        ret = BasicBlock(self.next_unique_name(name_prefix if name_prefix is not None else 'label.'))
        self.blocks.append(ret)
        return BasicBlockBuilder(self, ret)

class Inst:
    @abstractmethod
    def get_type(self) -> Type:
        pass

class BasicBlock(Value):
    name: str
    insts: list[tuple[RegisterValue | None, Inst]]

    def __init__(self, name: str) -> None:
        self.name = name
        self.insts = []

    def write(self) -> list[str]:
        ret: list[str] = []
        for value, inst in self.insts:
            if value is None:
                ret.append(str(inst))
            else:
                ret.append(f"%{value.name} = {inst}")
        return ret

    @override
    def get_type(self) -> Type:
        return LabelType()

    def __str__(self) -> str:
        return 'label %' + self.name

@dataclass
class BasicBlockBuilder:
    fn: Function
    block: BasicBlock

    def emit(self, inst: Inst, reg_name: str | None = None) -> Value:
        type = inst.get_type()
        match type:
            case VoidType():
                self.block.insts.append((None, inst))
                return VoidValue()
            case _:
                value = RegisterValue(self.fn.next_unique_name(reg_name if reg_name is not None else 'reg.'), type)
                self.block.insts.append((value, inst))
                return value

    def add(self, lhs: Value, rhs: Value, reg_name: str | None = None) -> Value:
        if isinstance(lhs, IntValue) and isinstance(rhs, IntValue):
            assert lhs.type == rhs.type
            return IntValue(lhs.value + rhs.value, lhs.type)
        if isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
            return FloatValue(lhs.value + rhs.value, lhs.type)

        match lhs.get_type():
            case IntType():
                if isinstance(rhs, IntValue) and rhs.value < 0:
                    return self.emit(Sub(lhs, IntValue(-rhs.value, rhs.type)), reg_name)
                return self.emit(Add(lhs, rhs), reg_name)
            case FloatType():
                return self.emit(FAdd(lhs, rhs), reg_name)
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def sub(self, lhs: Value, rhs: Value, reg_name: str | None = None) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Sub(lhs, rhs), reg_name)
            case FloatType():
                return self.emit(FSub(lhs, rhs), reg_name)
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def mul(self, lhs: Value, rhs: Value, reg_name: str | None = None) -> Value:
        if isinstance(lhs, IntValue) and isinstance(rhs, IntValue):
            assert lhs.type == rhs.type
            return IntValue(lhs.value * rhs.value, lhs.type)
        if isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
            return FloatValue(lhs.value * rhs.value, lhs.type)

        match lhs.get_type():
            case IntType():
                return self.emit(Mul(lhs, rhs), reg_name)
            case FloatType():
                return self.emit(FMul(lhs, rhs), reg_name)
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def div(self, lhs: Value, rhs: Value, reg_name: str | None = None) -> Value:
        if isinstance(lhs, IntValue) and isinstance(rhs, IntValue):
            return IntValue(lhs.value // rhs.value, lhs.type)
        if isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
            return FloatValue(lhs.value / rhs.value, lhs.type)

        match lhs.get_type():
            case IntType():
                return self.emit(Div(lhs, rhs), reg_name)
            case FloatType():
                return self.emit(FDiv(lhs, rhs), reg_name)
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def rem(self, lhs: Value, rhs: Value, reg_name: str | None = None) -> Value:
        if isinstance(lhs, IntValue) and isinstance(rhs, IntValue):
            return IntValue(lhs.value % rhs.value, lhs.type)
        if isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
            return FloatValue(lhs.value % rhs.value, lhs.type)

        match lhs.get_type():
            case IntType():
                return self.emit(Rem(lhs, rhs), reg_name)
            case FloatType():
                return self.emit(FRem(lhs, rhs), reg_name)
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def load(self, ptr: Value, reg_name: str | None = None) -> Value:
        return self.emit(Load(ptr), reg_name)

    def store(self, ptr: Value, value: Value):
        self.emit(Store(ptr, value))

    def ret(self, value: Value | None = None):
        self.emit(Ret(value if value is not None else VoidValue()))

    def coerce(self, value: Value, type: Type, reg_name: str | None = None) -> Value:
        value_type = value.get_type()
        if value_type == type:
            return value
        if isinstance(value_type, FloatType):
            if isinstance(type, FloatType):
                if value_type.bits() > type.bits():
                    if isinstance(value, FloatValue):
                        return FloatValue(value.value, type)
                    return self.emit(FloatTrunc(value, type), reg_name)
                else:
                    return self.emit(FloatExt(value, type), reg_name)
            if isinstance(type, IntType):
                return self.emit(FloatToInt(value, type), reg_name)
        if isinstance(value_type, IntType):
            if isinstance(type, FloatType):
                return self.emit(IntToFloat(value, type), reg_name)
            if isinstance(type, IntType):
                if value_type.bits > type.bits:
                    return self.emit(IntTrunc(value, type), reg_name)
                return self.emit(IntExt(value, type), reg_name)
        raise ValueError(f"Cannot coerce {value_type} to {type}")

    def get_array_element_ptr(self, array: Value, index: Value, reg_name: str | None = None) -> Value:
        assert isinstance(array.get_type(), PointerType)
        assert isinstance(index.get_type(), IntType)
        return self.emit(GetElementPtr(array, (index, )), reg_name)

    def sqrt(self, value: Value, reg_name: str | None = None) -> Value:
        match value:
            case FloatValue(fv, type):
                return FloatValue(math.sqrt(fv), type)
        match value.get_type():
            case Float32Type():
                return self.emit(Call(SQRT_F32, (value, )), reg_name)
            case Float64Type():
                return self.emit(Call(SQRT_F64, (value, )), reg_name)
            case _:
                raise TypeError(f"Cannot take sqrt of {value.get_type()}")

    def pow(self, value: Value, exponent: Value, reg_name: str | None = None) -> Value:
        if isinstance(value, FloatValue) and isinstance(exponent, FloatValue):
            return FloatValue(math.pow(value.value, exponent.value), value.type)

        match value.get_type():
            case Float32Type():
                return self.emit(Call(POW_F32, (value, exponent)), reg_name)
            case Float64Type():
                return self.emit(Call(POW_F64, (value, exponent)), reg_name)
            case _:
                raise TypeError(f"Cannot take pow of {value.get_type()}")

    def exp(self, value: Value, reg_name: str | None = None) -> Value:
        if isinstance(value, FloatValue):
            return FloatValue(math.exp(value.value), value.type)

        match value.get_type():
            case Float32Type():
                return self.emit(Call(EXP_F32, (value, )), reg_name)
            case Float64Type():
                return self.emit(Call(EXP_F64, (value, )), reg_name)
            case _:
                raise TypeError(f"Cannot take exp of {value.get_type()}")

    def sin(self, value: Value, reg_name: str | None = None) -> Value:
        if isinstance(value, FloatValue):
            return FloatValue(math.sin(value.value), value.type)

        match value.get_type():
            case Float32Type():
                return self.emit(Call(SIN_F32, (value, )), reg_name)
            case Float64Type():
                return self.emit(Call(SIN_F64, (value, )), reg_name)
            case _:
                raise TypeError(f"Cannot take sin of {value.get_type()}")

    def cos(self, value: Value, reg_name: str | None = None) -> Value:
        if isinstance(value, FloatValue):
            return FloatValue(math.cos(value.value), value.type)

        match value.get_type():
            case Float32Type():
                return self.emit(Call(COS_F32, (value, )), reg_name)
            case Float64Type():
                return self.emit(Call(COS_F64, (value, )), reg_name)
            case _:
                raise TypeError(f"Cannot take cos of {value.get_type()}")

    def ln(self, value: Value, reg_name: str | None = None) -> Value:
        if isinstance(value, FloatValue):
            return FloatValue(math.log(value.value), value.type)

        match value.get_type():
            case Float32Type():
                return self.emit(Call(LN_F32, (value, )), reg_name)
            case Float64Type():
                return self.emit(Call(LN_F64, (value, )), reg_name)
            case _:
                raise TypeError(f"Cannot take ln of {value.get_type()}")

@dataclass
class Binary(Inst):
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        assert self.lhs.get_type() == self.rhs.get_type()

    def __str__(self) -> str:
        return f'{self.head_name()} {self.lhs.get_type()} {self.lhs}, {self.rhs}'

    @override
    def get_type(self) -> Type:
        return self.lhs.get_type()

    @abstractmethod
    def head_name(self) -> str:
        pass

@dataclass
class Add(Binary):
    @override
    def head_name(self) -> str:
        return 'add'

@dataclass
class FAdd(Binary):
    @override
    def head_name(self) -> str:
        return 'fadd'

@dataclass
class Sub(Binary):
    @override
    def head_name(self) -> str:
        return 'sub'

@dataclass
class Mul(Binary):
    @override
    def head_name(self) -> str:
        return 'mul'

@dataclass
class FMul(Binary):
    @override
    def head_name(self) -> str:
        return 'fmul'

@dataclass
class FSub(Binary):
    @override
    def head_name(self) -> str:
        return 'fsub'

@dataclass
class Rem(Binary):
    @override
    def head_name(self) -> str:
        return 'rem'

@dataclass
class FRem(Binary):
    @override
    def head_name(self) -> str:
        return 'frem'

@dataclass
class Div(Binary):
    @override
    def head_name(self) -> str:
        return 'div'

@dataclass
class FDiv(Binary):
    @override
    def head_name(self) -> str:
        return 'fdiv'

@dataclass
class Load(Inst):
    ptr: Value
    type: Type = field(init=False, compare=False)

    def __post_init__(self):
        type = self.ptr.get_type()
        assert isinstance(type, PointerType), f"pointer type expected, got {type}"
        self.type = type

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'load {self.type}, {self.ptr}'

@dataclass
class FloatExt(Inst):
    value: Value
    type: FloatType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'fpext {self.type} {self.value} to {self.type}'

@dataclass
class FloatTrunc(Inst):
    value: Value
    type: FloatType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'fptrunc {self.type} {self.value} to {self.type}'

@dataclass
class FloatToInt(Inst):
    value: Value
    type: IntType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'fpto{'s' if self.type.signed else 'u'}i {self.type} {self.value} to {self.type}'

@dataclass
class IntToFloat(Inst):
    value: Value
    type: FloatType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        type = self.value.get_type()
        assert isinstance(type, IntType)
        return f'fpto{'u' if type.signed else 's'}i {type} {self.value} to {self.type}'

@dataclass
class IntTrunc(Inst):
    value: Value
    type: IntType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        type = self.value.get_type()
        assert isinstance(type, IntType)
        return f'trunc {type} {self.value} to {self.type}'

@dataclass
class IntExt(Inst):
    value: Value
    type: IntType

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        type = self.value.get_type()
        assert isinstance(type, IntType)
        return f'{'s' if self.type.signed else 'z'}ext {type} {self.value} to {self.type}'

@dataclass
class GetElementPtr(Inst):
    ptr: Value
    indices: tuple[Value, ...]
    type: Type = field(init=False)

    def __post_init__(self):
        type = self.ptr.get_type()
        assert isinstance(type, PointerType)
        self.type = type.child

    @override
    def get_type(self) -> Type:
        return self.ptr.get_type()

    def __str__(self) -> str:
        return f'getelementptr inbounds {type} {self.ptr}, {", ".join(str(i) for i in self.indices)}'

@dataclass
class Call(Inst):
    func: Value
    args: tuple[Value, ...]
    type: Type = field(init=False)

    def __post_init__(self):
        fn = self.func.get_type()
        if not isinstance(fn, FnType):
            raise TypeError(f'expected function type, got {fn}')
        if len(self.args) != len(fn.args):
            raise TypeError(f'expected {len(fn.args)} arguments, got {len(self.args)}')
        for arg, expected in zip(self.args, fn.args):
            if arg.get_type() != expected:
                raise TypeError(f'expected {expected}, got {arg.get_type()}')
        self.type = fn.return_type

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f'call {self.type} {self.func}({", ".join(str(arg) for arg in self.args)})'

@dataclass
class Store(Inst):
    ptr: Value
    value: Value

    def __post_init__(self):
        ptr_type = self.ptr.get_type()
        assert isinstance(ptr_type, PointerType), "pointer type expected"
        assert self.value.get_type() == ptr_type.child

    @override
    def get_type(self) -> Type:
        return VoidType()

    def __str__(self) -> str:
        return f'store {self.value}, {self.ptr}'

@dataclass
class Ret(Inst):
    value: Value

    @override
    def get_type(self) -> Type:
        return VoidType()

    def __str__(self) -> str:
        return f'ret {self.value}'


SQRT_F32 = DeclareFunction('llvm.sqrt.f32', FnType((Float32Type(),), Float32Type()))
SQRT_F64 = DeclareFunction('llvm.sqrt.f64', FnType((Float64Type(),), Float64Type()))
POW_F32 = DeclareFunction('llvm.pow.f32', FnType((Float32Type(), Float32Type()), Float32Type()))
POW_F64 = DeclareFunction('llvm.pow.f64', FnType((Float64Type(), Float64Type()), Float64Type()))
EXP_F32 = DeclareFunction('llvm.exp.f32', FnType((Float32Type(),), Float32Type()))
EXP_F64 = DeclareFunction('llvm.exp.f64', FnType((Float64Type(),), Float64Type()))
LN_F32 = DeclareFunction('llvm.log.f32', FnType((Float32Type(),), Float32Type()))
LN_F64 = DeclareFunction('llvm.log.f64', FnType((Float64Type(),), Float64Type()))
SIN_F32 = DeclareFunction('llvm.sin.f32', FnType((Float32Type(),), Float32Type()))
SIN_F64 = DeclareFunction('llvm.sin.f64', FnType((Float64Type(),), Float64Type()))
COS_F32 = DeclareFunction('llvm.cos.f32', FnType((Float32Type(),), Float32Type()))
COS_F64 = DeclareFunction('llvm.cos.f64', FnType((Float64Type(),), Float64Type()))
