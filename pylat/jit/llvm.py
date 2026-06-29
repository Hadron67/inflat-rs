from enum import Enum, IntEnum
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import final, override

from ..util import ObjectCounter, StrBiMap, SubExprFnBuilder

class NameContext:
    @abstractmethod
    def get_struct_type_name(self, type: 'StructType') -> str:
        pass

    @abstractmethod
    def get_global_name(self, value: 'GlobalValue') -> str:
        pass

class Type:
    def from_float(self, value: float) -> 'Value':
        raise TypeError(f'cannot create float from {self}')

    def from_int(self, value: int) -> 'Value':
        return self.from_float(float(value))

    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        return str(self)

    def get_children(self) -> 'list[Type]':
        return []

class Value:
    @abstractmethod
    def get_type(self) -> Type:
        pass

    @final
    def stringify(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        return self.get_type().stringify(name_context) + ' ' + self.stringify_value(name_context, local_counter)

    @abstractmethod
    def stringify_value(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        return str(self)

    def get_children(self) -> 'list[Value]':
        return []

def gen_get_children(cls=None, excludes: set[str] | None = None):
    def wrapper(cls):
        children_builder = SubExprFnBuilder(Value if issubclass(cls, Value) else Type)
        get_children = 'get_children'
        globals = {}
        source = '\n'.join(children_builder.generate_get_children(cls, get_children, excludes))
        exec(source, globals=globals)
        setattr(cls, get_children, globals[get_children])
        return cls

    if cls is not None:
        return wrapper(cls)
    return wrapper

class FloatType(Type):
    @abstractmethod
    def bits(self) -> int:
        pass

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class IntType(Type):
    bits: int

    def __str__(self) -> str:
        return f'{"i"}{self.bits}'

    @final
    def get_range(self, signed: bool) -> tuple[int, int]:
        return IntType.get_range_from_data(self.bits, signed)

    @staticmethod
    def get_range_from_data(bits: int, signed: bool):
        if signed:
            return (-2**(bits - 1), 2**(bits - 1) - 1)
        else:
            return (0, 2**bits - 1)

    @override
    def from_int(self, value: int) -> 'Value':
        min, max = self.get_range(True)
        if not min <= value <= max:
            raise ValueError(f'value {value} out of range for {self}')
        return IntValue(value, self)

I8 = IntType(8)
I32 = IntType(32)
I64 = IntType(64)

@dataclass(frozen=True)
class LabelType(Type):
    def __str__(self) -> str:
        return 'label'

@dataclass(frozen=True)
@gen_get_children
class PointerType(Type):
    child: Type

    def __str__(self) -> str:
        return f"*{self.child}"

    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        return 'ptr'

@dataclass(frozen=True)
@gen_get_children
class ArrayType(Type):
    child: Type
    length: int

    def __str__(self) -> str:
        return f"[{self.length} x {self.child}]"

    @override
    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        return f"[{self.length} x {self.child.stringify(name_context)}]"

@dataclass(frozen=True)
@gen_get_children
class FnType(Type):
    args: tuple[Type, ...]
    return_type: Type
    varargs: bool = False

    def __str__(self) -> str:
        return f'fn({",".join(str(arg) for arg in self.args)}) -> {self.return_type}'

    @override
    def get_children(self) -> 'list[Type]':
        ret = list(self.args)
        ret.append(self.return_type)
        return ret

@dataclass
class VoidType(Type):
    def __str__(self) -> str:
        return 'void'

@gen_get_children
class StructType(Type):
    fields: list[Type]

    @override
    def __init__(self, *fields: Type) -> None:
        self.fields = list(fields)

    def add_field(self, type: Type):
        self.fields.append(type)

    @override
    def __str__(self) -> str:
        return f"struct@{id(self)} {{{", ".join(str(i) for i in self.fields)}}}"

    def write_definition(self, name_context: 'NameContext') -> list[str]:
        return [f"%{name_context.get_struct_type_name(self)} = type {{{', '.join(i.stringify(name_context) for i in self.fields)}}}"]

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)

    @override
    def __eq__(self, value: object, /) -> bool:
        return self is value

    @override
    def stringify(self, name_context: 'NameContext | None' = None) -> str:
        if name_context is not None:
            return '%' + name_context.get_struct_type_name(self)
        else:
            return str(self)

class LocalValue(Value):
    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    @final
    @override
    def stringify_value(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        assert local_counter is not None
        return f"%{local_counter.get_id(self)}"

class ArgValue(LocalValue):
    type: Type
    index: int

    def __init__(self, type: Type, index: int) -> None:
        self.type = type
        self.index = index

    @override
    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} %{self.index}"

    def __repr__(self) -> str:
        return f"ArgValue@{id(self)}(type={repr(self.type)}, index={self.index})"

@dataclass
class VoidValue(Value):
    def __str__(self) -> str:
        return ''

    @override
    def get_type(self) -> Type:
        return VoidType()

@dataclass
class IntValue(Value):
    value: int
    type: IntType

    def __str__(self) -> str:
        return str(self.value)

    @override
    def get_type(self) -> Type:
        return self.type

BOOL_TYPE = IntType(1)
BOOL_TRUE = IntValue(1, BOOL_TYPE)
BOOL_FALSE = IntValue(0, BOOL_TYPE)

@dataclass
class FloatValue(Value):
    value: float
    type: FloatType

    def __str__(self) -> str:
        return str(self.value)

    @override
    def get_type(self) -> Type:
        return self.type

@dataclass
# @gen_get_children
class AggregateValue(Value):
    type: Type
    values: tuple[Value, ...]

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_value(self, name_context: NameContext, local_counter: 'ObjectCounter[LocalValue] | None' = None) -> str:
        return f"{{{", ".join(v.stringify(name_context, local_counter) for v in self.values)}}}"

_CHAR_CODES = [

]

@dataclass
class StringLiteralValue(Value):
    data: bytes

    @override
    def get_type(self) -> Type:
        return ArrayType(I8, len(self.data))

    @override
    def __str__(self) -> str:
        ret = 'c"'
        for d in self.data:
            # TODO: escape all unprintable chars
            if d == 0:
                ret += '\\00'
            else:
                ret += bytes([d]).decode()
        ret += '"'
        return ret

class Module(NameContext):
    _globals: 'StrBiMap[GlobalValue]'
    _struct_types: 'StrBiMap[StructType]'

    def __init__(self) -> None:
        self._globals = StrBiMap()
        self._struct_types = StrBiMap()

    def add_recursively(self, types: list[Type] | None = None, values: 'list[Value] | None' = None):
        todo_values: list[Value] = []
        todo_types: list[Type] = []
        if types is not None:
            todo_types.extend(types)
        if values is not None:
            todo_values.extend(values)

        while len(todo_values) > 0:
            value = todo_values.pop()
            assert not isinstance(value, Inst)
            if isinstance(value, GlobalValue):
                if self._globals.has_value(value):
                    continue
                name = value.get_required_name()
                assert name is None or not self._globals.has_key(name)
                self._add_global(value, name)
            todo_values.extend(value.get_children())
            todo_types.append(value.get_type())

            while len(todo_types) > 0:
                type = todo_types.pop()
                if isinstance(type, StructType):
                    if self._struct_types.has_value(type):
                        continue
                    self._add_struct_type(type)
                todo_types.extend(type.get_children())

    @override
    def get_global_name(self, value: 'GlobalValue') -> str:
        return self._globals.get_key(value)

    @override
    def get_struct_type_name(self, type: 'StructType') -> str:
        return self._struct_types.get_key(type)

    def _add_global(self, value: 'GlobalValue', name_prefix: str | None = None):
        if name_prefix is None:
            name_prefix = value.get_default_name_prefix()
        self._globals.add(self._globals.next_unique_name(name_prefix), value)

    def _add_struct_type(self, type: StructType, name_prefix: str | None = None):
        if name_prefix is None:
            name_prefix = 'struct'
        self._struct_types.add(self._struct_types.next_unique_name(name_prefix), type)

    def write(self) -> list[str]:
        ret: list[str] = []
        for type in self._struct_types.values():
            ret.extend(type.write_definition(self))
        for value in self._globals.values():
            ret.extend(value.write_definition(self))
        return ret

class GlobalValue(Value):
    def __eq__(self, value: object, /) -> bool:
        return self is value

    def __hash__(self) -> int:
        return object.__hash__(self)

    @abstractmethod
    def write_definition(self, name_context: NameContext) -> list[str]:
        pass

    @override
    @final
    def stringify_value(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue] | None = None) -> str:
        return '@' + name_context.get_global_name(self)

    def get_required_name(self) -> str | None:
        return None

    @abstractmethod
    def get_default_name_prefix(self) -> str:
        raise NotImplementedError

@gen_get_children
class GlobalDefineValue(GlobalValue):
    value: Value
    is_const: bool = True
    is_private: bool = True
    is_unnamed_addr: bool = True

    def __init__(self, value: Value, is_const: bool = True, is_private: bool= True, is_unnamed_addr: bool = True) -> None:
        self.value = value
        super().__init__()

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        flags: str = ''
        if self.is_private:
            flags += 'private '
        if self.is_unnamed_addr:
            flags += 'unnamed_addr '
        if self.is_const:
            flags += 'constant '
        return [f'@{name_context.get_global_name(self)} = {flags}{self.value.stringify(name_context)}']

    @override
    def get_type(self) -> Type:
        return PointerType(self.value.get_type())

    @override
    def get_default_name_prefix(self):
        return "global"

@gen_get_children
class DeclareFunction(GlobalValue):
    type: FnType
    name: str

    def __init__(self, name: str, type: FnType, internal: bool = False):
        self.name = name
        self.type = type

    @override
    def get_type(self) -> Type:
        return PointerType(self.type)

    def write_args(self, name_context: NameContext) -> list[str]:
        args: list[str] = []
        for arg in self.type.args:
            args.append(arg.stringify(name_context))
        if self.type.varargs:
            args.append('...')
        return args

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        return [f"declare {self.type.return_type.stringify(name_context)} @{self.name}({', '.join(self.write_args(name_context))})"]

    def get_required_name(self) -> str | None:
        return self.name

    @override
    def get_default_name_prefix(self):
        return 'fn'

class FunctionState(IntEnum):
    BUILDING_ARGS = 0
    BUILDING_RETURN_TYPE = 1
    BUILDING_BODY = 2
    FINISHED = 3

class IFunction:
    @abstractmethod
    def add_arg(self, type: Type) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_arg(self, index: int) -> Value:
        raise NotImplementedError

class FunctionArgs(IFunction):
    parent: 'FunctionArgs | None'
    args: list[ArgValue]

    def __init__(self, parent: 'FunctionArgs | None' = None) -> None:
        self.parent = parent
        self.args = []

    @override
    def get_arg(self, index: int) -> Value:
        return self.args[index]

    @override
    def add_arg(self, type: Type) -> int:
        ret = len(self.args)
        self.args.append(ArgValue(type, ret))
        return ret

class Function(GlobalValue, IFunction):
    name: str | None
    _entry: 'BasicBlock'
    _args: list[ArgValue]

    _type: FnType | None

    def __init__(self, name: str | None = None, entry: 'BasicBlock | None' = None) -> None:
        self.name = name
        self._entry = entry if entry is not None else BasicBlock()
        self._args = []
        self._type = None

    @property
    def entry(self):
        return self._entry

    @override
    def write_definition(self, name_context: NameContext) -> list[str]:
        assert self._type is not None
        ret: list[str] = []
        local_counter: ObjectCounter[LocalValue] = ObjectCounter()
        ret.append(f'define {'internal ' if self.name is None else ''}{self._type.return_type.stringify(name_context)} @{name_context.get_global_name(self)}({', '.join(i.stringify(name_context, local_counter) for i in self._args)}) {{')

        blocks = self._entry.collect_blocks()

        # allocate the indices first, as required by llvm ir
        for block in blocks:
            local_counter.get_id(block)
            for inst in block.insts:
                if not isinstance(inst.get_type(), VoidType):
                    local_counter.get_id(inst)

        for block in blocks:
            ret.append(str(local_counter.get_id(block)) + ':')
            for line in block.write(name_context, local_counter):
                ret.append('    ' + line)
        ret.append('}')
        return ret

    @override
    def get_type(self) -> Type:
        assert self._type is not None
        return PointerType(self._type)

    @override
    def add_arg(self, type: Type) -> int:
        assert self._type is None or self._type.varargs
        ret = ArgValue(type, len(self._args))
        self._args.append(ret)
        return ret.index

    def add_args(self, *types: Type):
        return tuple(self.get_arg(self.add_arg(t)) for t in types)

    @override
    def get_arg(self, index: int) -> Value:
        return self._args[index]

    def set_return_type(self, type: Type, varargs: bool = False):
        assert self._type is None
        self._type = FnType(tuple(i.type for i in self._args), type, varargs)

    @override
    def get_children(self) -> list[Value]:
        assert self._type is not None
        # exclude all Inst when collecting

        ret: list[Value] = []
        for b in self._entry.collect_blocks():
            for inst in b.insts:
                ret.extend(i for i in inst.get_children() if not isinstance(i, Inst) and not isinstance(i, BasicBlock))
        return ret

    def get_required_name(self) -> str | None:
        return self.name

    @override
    def get_default_name_prefix(self) -> str:
        return 'fn'

class Inst(LocalValue):
    @abstractmethod
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        pass

    def try_evaluate(self) -> Value | None:
        return None

class Nop(Inst):
    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        raise RuntimeError("Nop in inst block")

class BasicBlock(LocalValue):
    insts: list[Inst]

    def __init__(self) -> None:
        self.insts = []

    @override
    def get_children(self) -> 'list[Value]':
        raise RuntimeError("cannot be called directly")

    def get_outgoing_blocks(self):
        ret: list[BasicBlock] = []
        seen_blocks: set[BasicBlock] = set()
        for inst in self.insts:
            for c in inst.get_children():
                if isinstance(c, BasicBlock) and c not in seen_blocks:
                    seen_blocks.add(c)
                    ret.append(c)
        return ret

    @final
    def write(self, name_context: NameContext, inst_counter: ObjectCounter[LocalValue]) -> list[str]:
        ret: list[str] = []
        for inst in self.insts:
            if isinstance(inst, Nop):
                continue
            if inst.get_type() == VoidType():
                ret.append(inst.stringify_inst(name_context, inst_counter))
            else:
                ret.append(f"%{inst_counter.get_id(inst)} = {inst.stringify_inst(name_context, inst_counter)}")
        return ret

    def collect_blocks(self):
        ret: list[BasicBlock] = []
        seen_blocks: set[BasicBlock] = set()
        todo: list[BasicBlock] = [self]
        while len(todo) > 0:
            item = todo.pop()
            if item in seen_blocks:
                continue
            seen_blocks.add(item)
            ret.append(item)
            children = item.get_outgoing_blocks()
            children.reverse()
            todo.extend(children)
        return ret

    @override
    def get_type(self) -> Type:
        return LabelType()

    # build methods

    def emit(self, inst: Inst) -> Value:
        value = inst.try_evaluate()
        if value is not None:
            return value
        self.insts.append(inst)
        return inst

    def reserve(self) -> int:
        ret = len(self.insts)
        self.emit(Nop())
        return ret

    def emit_at(self, index: int, inst: Inst):
        value = inst.try_evaluate()
        if value is not None:
            return value
        i = self.insts[index]
        assert isinstance(i, Nop), f"expected a nop, found {i}"
        self.insts[index] = inst

    def add(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                if isinstance(rhs, IntValue) and rhs.value < 0:
                    return self.emit(Sub(lhs, IntValue(-rhs.value, rhs.type)))
                return self.emit(Add(lhs, rhs))
            case FloatType():
                return self.emit(FAdd(lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def sub(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Sub(lhs, rhs))
            case FloatType():
                return self.emit(FSub(lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def mul(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Mul(lhs, rhs))
            case FloatType():
                return self.emit(FMul(lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def div(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Div(lhs, rhs))
            case FloatType():
                return self.emit(FDiv(lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def rem(self, lhs: Value, rhs: Value) -> Value:
        match lhs.get_type():
            case IntType():
                return self.emit(Rem(lhs, rhs))
            case FloatType():
                return self.emit(FRem(lhs, rhs))
            case _:
                raise TypeError(f'unsupported type: {lhs.get_type()}')

    def load(self, ptr: Value) -> Value:
        return self.emit(Load(ptr))

    def store(self, ptr: Value, value: Value | int):
        if isinstance(value, Value):
            self.emit(Store(ptr, value))
        else:
            ptr_type = ptr.get_type()
            assert isinstance(ptr_type, PointerType)
            assert isinstance(ptr_type.child, IntType)
            return self.emit(Store(ptr, IntValue(value, ptr_type.child)))

    def alloca(self, type: Type):
        return self.emit(Alloca(type))

    def ret(self, value: Value | None = None):
        self.emit(Ret(value if value is not None else VoidValue()))

    def extract_value(self, value: Value, *indices: int):
        return self.emit(ExtractValue(value, indices))

    def fneg(self, value: Value):
        return self.emit(FNeg(value))

    def int_to_float(self, value: Value, float_type: FloatType):
        return self.emit(IntToFloat(value, float_type))

    def get_element_ptr(self, array: Value, *indices: Value) -> Value:
        return self.emit(GetElementPtr(array, indices))

    def icmp(self, op: 'IcmpOp', signed: bool, lhs: Value, rhs: Value):
        return self.emit(Icmp(op, signed, lhs, rhs))

    def phi(self, *incomings: 'tuple[Value, BasicBlock]'):
        ret = self.emit(Phi(*incomings))
        assert isinstance(ret, Phi)
        return ret

    def br(self, cond: Value, if_true: 'BasicBlock', if_false: 'BasicBlock'):
        self.emit(Br(cond, if_true, if_false))

    def jmp(self, target: 'BasicBlock'):
        self.emit(BrDirect(target))

    def call(self, fn: Value, *args: Value):
        return self.emit(Call(fn, args))

    def sqrt(self, value: Value, reg_name: str | None = None) -> Value:
        match value:
            case FloatValue(fv, type):
                return FloatValue(math.sqrt(fv), type)
        match value.get_type():
            case Float32Type():
                return self.call(SQRT_F32, value)
            case Float64Type():
                return self.call(SQRT_F64, value)
            case _:
                raise TypeError(f"Cannot take sqrt of {value.get_type()}")

    def pow(self, value: Value, exponent: Value) -> Value:
        match value.get_type():
            case Float32Type():
                return self.call(POW_F32, value, exponent)
            case Float64Type():
                return self.call(POW_F64, value, exponent)
            case _:
                raise TypeError(f"Cannot take pow of {value.get_type()}")

    def exp(self, value: Value) -> Value:
        match value.get_type():
            case Float32Type():
                return self.call(EXP_F32, value)
            case Float64Type():
                return self.call(EXP_F64, value)
            case _:
                raise TypeError(f"Cannot take exp of {value.get_type()}")

    def sin(self, value: Value) -> Value:
        match value.get_type():
            case Float32Type():
                return self.call(SIN_F32, value)
            case Float64Type():
                return self.call(SIN_F64, value)
            case _:
                raise TypeError(f"Cannot take sin of {value.get_type()}")

    def cos(self, value: Value) -> Value:
        match value.get_type():
            case Float32Type():
                return self.call(COS_F32, value)
            case Float64Type():
                return self.call(COS_F64, value)
            case _:
                raise TypeError(f"Cannot take cos of {value.get_type()}")

    def ln(self, value: Value) -> Value:
        match value.get_type():
            case Float32Type():
                return self.call(LN_F32, value)
            case Float64Type():
                return self.call(LN_F64, value)
            case _:
                raise TypeError(f"Cannot take ln of {value.get_type()}")

@gen_get_children
class Unary(Inst):
    child: Value

    def __init__(self, child: Value) -> None:
        self.child = child

    @abstractmethod
    def _check(self):
        pass

    @abstractmethod
    def head_name(self) -> str:
        pass

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f"{self.head_name()} {self.child.stringify(name_context, local_counter)}"

    def get_type(self) -> Type:
        return self.child.get_type()

class FNeg(Unary):
    @override
    def head_name(self) -> str:
        return "fneg"

    @override
    def _check(self):
        type = self.child.get_type()
        assert isinstance(type, FloatType), f"float type expected, got {type}"

    def try_evaluate(self) -> Value | None:
        match self.child:
            case FloatValue(value, t):
                return FloatValue(-value, t)
        return None

@gen_get_children
class Binary(Inst):
    lhs: Value
    rhs: Value
    type: Type

    def __init__(self, lhs: Value, rhs: Value) -> None:
        self.lhs = lhs
        self.rhs = rhs
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert lhs_type == rhs_type, f"incompatible types {lhs_type} and {rhs_type}"
        self.type = lhs_type

    @abstractmethod
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'{self.head_name()} {self.type} {self.lhs.stringify_value(name_context, local_counter)}, {self.rhs.stringify_value(name_context, local_counter)}'

    @override
    def get_type(self) -> Type:
        return self.lhs.get_type()

    @abstractmethod
    def head_name(self) -> str:
        pass

class Add(Binary):
    @override
    def head_name(self) -> str:
        return 'add'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a + b, t)
        return None

class FAdd(Binary):
    @override
    def head_name(self) -> str:
        return 'fadd'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a + b, t)
            case FloatValue(0, t), _:
                return self.rhs
            case _, FloatValue(0, t):
                return self.lhs
        return None

class Sub(Binary):
    @override
    def head_name(self) -> str:
        return 'sub'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a - b, t)
        return None

class Mul(Binary):
    @override
    def head_name(self) -> str:
        return 'mul'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a * b, t)
        return None

class FMul(Binary):
    @override
    def head_name(self) -> str:
        return 'fmul'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a * b, t)
            case FloatValue(0, t), _:
                return FloatValue(0, t)
            case _, FloatValue(0, t):
                return FloatValue(0, t)
        return None

class FSub(Binary):
    @override
    def head_name(self) -> str:
        return 'fsub'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a - b, t)
            case _, FloatValue(0, _):
                return self.lhs
        return None

class Rem(Binary):
    @override
    def head_name(self) -> str:
        return 'rem'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a % b, t)
        return None

class FRem(Binary):
    @override
    def head_name(self) -> str:
        return 'frem'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(math.fmod(a, b), t)
        return None

class Div(Binary):
    @override
    def head_name(self) -> str:
        return 'div'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case IntValue(a, t), IntValue(b, _):
                return IntValue(a // b, t)
        return None

class FDiv(Binary):
    @override
    def head_name(self) -> str:
        return 'fdiv'

    @override
    def try_evaluate(self) -> Value | None:
        match self.lhs, self.rhs:
            case FloatValue(a, t), FloatValue(b, _):
                return FloatValue(a / b, t)
        return None

@gen_get_children
class Load(Inst):
    ptr: Value
    type: Type

    def __init__(self, ptr: Value):
        self.ptr = ptr
        type = ptr.get_type()
        assert isinstance(type, PointerType), f"pointer type expected, got {type}"
        self.type = type.child

    @override
    def get_type(self) -> Type:
        return self.type

    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'load {self.type.stringify(name_context)}, {self.ptr.stringify(name_context, local_counter)}'

@gen_get_children
class Alloca(Inst):
    type: Type

    @override
    def __init__(self, type: Type) -> None:
        self.type = type

    @override
    def get_type(self) -> Type:
        return PointerType(self.type)

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        type = self.type.stringify(name_context)
        return f"alloca {type}"

@gen_get_children
class ConversionInst[T: Type](Inst):
    value: Value
    type: T

    def __init__(self, value: Value, type: T) -> None:
        self.value = value
        self.type = type

    @abstractmethod
    def head_name(self) -> str:
        pass

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'{self.head_name()} {self.value.stringify(name_context, local_counter)} to {self.type.stringify(name_context)}'

    @override
    def get_type(self) -> Type:
        return self.type

class FloatExt(ConversionInst[FloatType]):
    @override
    def head_name(self) -> str:
        return 'fpext'

class FloatTrunc(ConversionInst[FloatType]):
    def head_name(self) -> str:
        return 'fptrunc'

class FloatToInt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'fptosi'

class FloatToUInt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'fptoui'

class IntToFloat(ConversionInst[FloatType]):
    @override
    def head_name(self) -> str:
        return 'fptosi'

    def try_evaluate(self) -> Value | None:
        if isinstance(self.value, IntValue):
            return FloatValue(float(self.value.value), self.type)
        return None

class UIntToFloat(ConversionInst[FloatType]):
    @override
    def head_name(self) -> str:
        return 'fptoui'

    def try_evaluate(self) -> Value | None:
        if isinstance(self.value, IntValue):
            return FloatValue(float(self.value.value), self.type)
        return None

class IntTrunc(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'trunc'

class IntExt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'sext'

class UIntExt(ConversionInst[IntType]):
    @override
    def head_name(self) -> str:
        return 'zext'

@gen_get_children
class GetElementPtr(Inst):
    ptr: Value
    indices: tuple[Value, ...]
    type: PointerType

    def __init__(self, ptr: Value, indices: tuple[Value, ...]):
        self.ptr = ptr
        self.indices = indices
        type = self.ptr.get_type()
        assert isinstance(type, PointerType), "expected pointer type"
        type = type.child
        i0 = indices[0]
        i0_type = i0.get_type()
        assert isinstance(i0_type, IntType) and i0_type.bits == 64, f"i64 is required for array subscripts, got {i0_type}"
        for i in indices[1:]:
            match type:
                case StructType():
                    assert isinstance(i, IntValue) and i.type.bits == 32, f"i32 constant is required for struct index, got {i}"
                    assert i.value < len(type.fields), f"struct index {i.value} is out of bounds for type {type}"
                    type = type.fields[i.value]
                case ArrayType():
                    i_type = i.get_type()
                    assert isinstance(i_type, IntType) and i_type.bits >= 32, f"i32 or i64 is required for array index, got {i_type}"
                    type = type.child
                case _:
                    raise TypeError(f"cannot get element of type {type}")
        self.type = PointerType(type)

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'getelementptr inbounds {self.type.child.stringify(name_context)}, {self.ptr.stringify(name_context, local_counter)}, {', '.join(i.stringify(name_context, local_counter) for i in self.indices)}'

@gen_get_children
class Call(Inst):
    fn: Value
    args: tuple[Value, ...]
    type: Type

    def __init__(self, fn: Value, args: tuple[Value, ...]):
        self.fn = fn
        self.args = args
        fn_ptr_type = fn.get_type()
        assert isinstance(fn_ptr_type, PointerType), "expected pointer type"
        fn_type = fn_ptr_type.child
        assert isinstance(fn_type, FnType), "expected function type"
        if not fn_type.varargs:
            assert len(args) == len(fn_type.args), "argument mismatch"
        else:
            assert len(args) >= len(fn_type.args), "argument mismatch"
        for i in range(len(fn_type.args)):
            arg = args[i]
            expected = fn_type.args[i]
            arg_type = arg.get_type()
            assert arg_type == expected, f"argument type mismatch at {i}-th arg: {arg_type} != {expected}"
        self.type = fn_type.return_type

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'call {self.type.stringify(name_context)} {self.fn.stringify_value(name_context, local_counter)}({', '.join(arg.stringify(name_context, local_counter) for arg in self.args)})'

@gen_get_children
class Store(Inst):
    ptr: Value
    value: Value

    def __init__(self, ptr: Value, value: Value):
        self.ptr = ptr
        self.value = value
        ptr_type = ptr.get_type()
        assert isinstance(ptr_type, PointerType), "pointer type expected"
        assert self.value.get_type() == ptr_type.child

    @override
    def get_type(self) -> Type:
        return VoidType()

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'store {self.value.stringify(name_context, local_counter)}, {self.ptr.stringify(name_context, local_counter)}'

@gen_get_children
class Ret(Inst):
    value: Value

    def __init__(self, value: Value) -> None:
        self.value = value

    @override
    def get_type(self) -> Type:
        return VoidType()

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f'ret {self.value.stringify(name_context, local_counter)}'

@gen_get_children
class ExtractValue(Inst):
    value: Value
    indices: tuple[int, ...]
    type: Type

    def __init__(self, value: Value, indices: tuple[int, ...]) -> None:
        self.value = value
        self.indices = indices
        type = value.get_type()
        assert len(indices) > 0
        for i in indices:
            match type:
                case StructType():
                    assert i < len(type.fields), f"index {i} is out of bounds for type {type}"
                    type = type.fields[i]
                case ArrayType():
                    assert i < type.length, f"index {i} is out of bounds for type {type}"
                    type = type.child
                case _:
                    raise TypeError(f"cannot extract value from type {type}")
        self.type = type

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f"extractvalue {self.value.stringify(name_context, local_counter)}, {', '.join(str(i) for i in self.indices)}"

    @override
    def try_evaluate(self) -> Value | None:
        value = self.value
        for i in self.indices:
            if not isinstance(value, AggregateValue):
                return None
            value = value.values[i]
        return value

@gen_get_children
class Br(Inst):
    condition: Value
    if_true: BasicBlock
    if_false: BasicBlock

    def __init__(self, cond: Value, if_true: BasicBlock, if_false: BasicBlock) -> None:
        self.condition = cond
        self.if_true = if_true
        self.if_false = if_false
        cond_type = cond.get_type()
        assert cond_type == BOOL_TYPE, f"bool type expected, got {cond_type}"

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        cond = self.condition.stringify(name_context, local_counter)
        if_true = self.if_true.stringify(name_context, local_counter)
        if_false = self.if_false.stringify(name_context, local_counter)
        return f"br {cond}, {if_true}, {if_false}"

    @override
    def get_type(self) -> Type:
        return VoidType()

@gen_get_children
class BrDirect(Inst):
    target: BasicBlock

    @override
    def __init__(self, target: BasicBlock) -> None:
        self.target = target

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        return f"br {self.target.stringify(name_context, local_counter)}"

    @override
    def get_type(self) -> Type:
        return VoidType()

@gen_get_children
class Phi(Inst):
    type: Type
    incomings: list[tuple[Value, BasicBlock]]

    @override
    def __init__(self, incoming: tuple[Value, BasicBlock]) -> None:
        self.type = incoming[0].get_type()
        self.incomings = [incoming]

    def add_incoming(self, value: Value, block: BasicBlock):
        type = value.get_type()
        assert self.type == type, f"types mismatch: {self.type} and {type}"
        self.incomings.append((value, block))

    @override
    def get_type(self) -> Type:
        return self.type

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        type = self.type.stringify(name_context)
        labels = (f"[{v.stringify_value(name_context, local_counter)}, {b.stringify_value(name_context, local_counter)}]" for v, b in self.incomings)
        return f"phi {type} {', '.join(labels)}"

class IcmpOp(Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"

@gen_get_children
class Icmp(Inst):
    type: IntType
    signed: bool
    op: IcmpOp
    lhs: Value
    rhs: Value

    @override
    def __init__(self, op: IcmpOp, signed: bool, lhs: Value, rhs: Value) -> None:
        lhs_type = lhs.get_type()
        rhs_type = rhs.get_type()
        assert isinstance(lhs_type, IntType), f"int type expected, got {lhs_type}"
        assert lhs_type == rhs_type, f"types mismatch: {lhs_type} and {rhs_type}"
        self.type = lhs_type
        self.signed = signed
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    @override
    def stringify_inst(self, name_context: NameContext, local_counter: ObjectCounter[LocalValue]) -> str:
        op: str = ''
        match self.op:
            case IcmpOp.EQ | IcmpOp.NE:
                op = self.op.name
            case _:
                op = ('s' if self.signed else 'u') + self.op.name
        type = self.type.stringify(name_context)
        lhs = self.lhs.stringify_value(name_context, local_counter)
        rhs = self.rhs.stringify_value(name_context, local_counter)
        return f"icmp {op} {type} {lhs}, {rhs}"

    @override
    def get_type(self) -> Type:
        return BOOL_TYPE

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
