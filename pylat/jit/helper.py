import ctypes
from dataclasses import dataclass
from typing import TypeAlias

from .argpass import TypesConfig, LowerType, ComplexFloatType
from .llvm import F64, I32, I64, I8, BasicBlock, DeclareFunction, FloatType, FloatValue, FnType, GlobalStringValue, IntType, PointerType, Value
from . import argpass as ap

@dataclass
class ComplexValue:
    re: Value
    im: Value

MaybeComplexValue: TypeAlias = ComplexValue | Value

class CompileHelper:
    parent: TypesConfig
    llvm_index_type: IntType
    llvm_real_type: FloatType

    def __init__(self, parent: TypesConfig) -> None:
        self.parent = parent
        self.llvm_index_type = IntType(parent.index_type.bits)
        self.llvm_real_type = FloatType(parent.real_type.bits)

    def promote_lower_type(self, type: ap.LowerType) -> LowerType:
        match type:
            case ap.IntType():
                return self.parent.index_type
            case ap.FloatType():
                return self.parent.real_type
            case ap.ComplexFloatType():
                return ComplexFloatType(self.parent.real_type)
            case _:
                raise TypeError(f"cannot promote lower type: {type}")

    def expand_complex_value(self, b: BasicBlock, value: MaybeComplexValue) -> tuple[Value, Value]:
        match value:
            case ComplexValue(re, im):
                return re, im
            case _:
                return b.extract_value(value, 0), b.extract_value(value, 1)

    def complex_add(self, block: BasicBlock, a: MaybeComplexValue, b: MaybeComplexValue) -> ComplexValue:
        a_re, a_im = self.expand_complex_value(block, a)
        b_re, b_im = self.expand_complex_value(block, b)
        return ComplexValue(
            block.add(a_re, b_re),
            block.add(a_im, b_im),
        )

    def complex_sub(self, block: BasicBlock, a: MaybeComplexValue, b: MaybeComplexValue) -> ComplexValue:
        a_re, a_im = self.expand_complex_value(block, a)
        b_re, b_im = self.expand_complex_value(block, b)
        return ComplexValue(
            block.sub(a_re, b_re),
            block.sub(a_im, b_im),
        )

    def complex_mul(self, block: BasicBlock, a: MaybeComplexValue, b: MaybeComplexValue) -> ComplexValue:
        a_re, a_im = self.expand_complex_value(block, a)
        b_re, b_im = self.expand_complex_value(block, b)
        return ComplexValue(
            block.sub(
                block.mul(a_re, b_re),
                block.mul(a_im, b_im),
            ),
            block.add(
                block.mul(a_re, b_im),
                block.mul(a_im, b_re),
            ),
        )

    def complex_div(self, block: BasicBlock, a: MaybeComplexValue, b: MaybeComplexValue):
        b_re, b_im = self.expand_complex_value(block, b)
        den = block.add(
            block.mul(b_re, b_re),
            block.mul(b_im, b_im),
        )
        b_re = block.div(b_re, den, True)
        b_im = block.fneg(block.div(b_im, den, True))
        return self.complex_mul(block, a, ComplexValue(b_re, b_im))

    def coerce_to_complex_type(self, block: BasicBlock, value: MaybeComplexValue, value_type: ap.LowerType, target_type: ap.ComplexFloatType) -> MaybeComplexValue:
        match value_type:
            case ap.ComplexFloatType(type):
                if target_type.type.bits > type.bits:
                    re, im = self.expand_complex_value(block, value)
                    return ComplexValue(self.coerce_to_real_type(block, re, type, target_type.type), self.coerce_to_real_type(block, im, type, target_type.type))
                if target_type == value_type:
                    return value
                raise TypeError(f"cannot coerce type {value_type} to {target_type}")
            case ap.FloatType():
                assert not isinstance(value, ComplexValue)
                return ComplexValue(value, target_type.type.to_llvm_value(0))
            case ap.IntType():
                assert not isinstance(value, ComplexValue)
                return ComplexValue(block.int_to_float(value, self.llvm_real_type), FloatValue(0, self.llvm_real_type))
            case _:
                raise TypeError(f"cannot coerce type {value_type} to complex")

    def coerce_to_real_type(self, block: BasicBlock, value: Value, value_type: ap.LowerType, target_type: ap.FloatType) -> Value:
        match value_type:
            case ap.FloatType():
                if target_type == value_type:
                    return value
                if target_type.bits > value_type.bits:
                    return block.float_ext(value, target_type.to_llvm_type())
                raise TypeError(f"cannot coerce type {value_type} to {target_type}")
            case ap.IntType(_, signed):
                if signed:
                    return block.int_to_float(value, target_type.to_llvm_type())
                else:
                    return block.uint_to_float(value, target_type.to_llvm_type())
            case _:
                raise TypeError(f"cannot coerce type {value_type} to real")

    def coerce(self, block: BasicBlock, value: MaybeComplexValue, value_type: ap.LowerType, target_type: ap.LowerType) -> MaybeComplexValue:
        match target_type:
            case ap.ComplexFloatType():
                return self.coerce_to_complex_type(block, value, value_type, target_type)
            case ap.FloatType():
                assert not isinstance(value, ComplexValue)
                return self.coerce_to_real_type(block, value, value_type, target_type)
            case ap.IntType():
                assert not isinstance(value, ComplexValue)
                assert isinstance(value_type, ap.IntType), f"expected IntType, got {value_type}"
                if target_type.bits > value_type.bits:
                    if target_type.signed:
                        return block.sext(value, target_type.to_llvm_type())
                    else:
                        return block.zext(value, target_type.to_llvm_type())
                if target_type == value_type:
                    return value
                raise TypeError(f"cannot coerce {value_type} to integer")
            case _:
                raise TypeError("????")

    def coerce_int_to_float(self, block: BasicBlock, value: MaybeComplexValue, value_type: LowerType, target_type: ap.FloatType) -> tuple[MaybeComplexValue, LowerType]:
        if isinstance(value_type, ap.IntType):
            assert not isinstance(value, ComplexValue)
            if value_type.signed:
                return block.int_to_float(value, target_type.to_llvm_type()), target_type
            else:
                return block.uint_to_float(value, target_type.to_llvm_type()), target_type
        else:
            return value, value_type

    def coerce_lower_type(self, block: BasicBlock, value: MaybeComplexValue, value_type: LowerType, target_type: LowerType) -> MaybeComplexValue:
        match target_type:
            case ComplexFloatType():
                assert isinstance(value_type, ComplexFloatType)
                if target_type.type.bits > value_type.type.bits:
                    re, im = self.expand_complex_value(block, value)
                    re = block.float_ext(re, FloatType(target_type.type.bits))
                    im = block.float_ext(im, FloatType(target_type.type.bits))
                    return ComplexValue(re, im)
                if target_type.type.bits < value_type.type.bits:
                    re, im = self.expand_complex_value(block, value)
                    re = block.float_ext(re, FloatType(target_type.type.bits))
                    im = block.float_ext(im, FloatType(target_type.type.bits))
                    return ComplexValue(re, im)
                return value
            case ap.FloatType():
                assert not isinstance(value, ComplexValue)
                assert isinstance(value_type, ap.FloatType)
                if target_type.bits > value_type.bits:
                    return block.float_ext(value, FloatType(value_type.bits))
                if target_type.bits < value_type.bits:
                    return block.float_trunc(value, FloatType(value_type.bits))
                return value
            case ap.IntType():
                assert isinstance(value_type, ap.IntType)
                assert target_type == value_type
                return value
        raise NotImplementedError

def echo(block: BasicBlock, *values: Value | str):
    for value in values:
        if isinstance(value, str):
            block.call(_ECHO_STR, GlobalStringValue(value.encode() + b'\0'))
            continue
        type = value.get_type()
        match type:
            case IntType(bits):
                match bits:
                    case 64:
                        block.call(_ECHO_I64, value)
                    case 32:
                        block.call(_ECHO_I32, value)
                    case _:
                        raise NotImplementedError
            case PointerType():
                block.call(_ECHO_PTR, value)
            case FloatType(64):
                block.call(_ECHO_F64, value)
            case _:
                raise NotImplementedError(f"TODO: type {type}")
    block.call(_ECHO_STR, GlobalStringValue(b'\n\0'))

_ECHO_STR = DeclareFunction('echo_str', FnType((PointerType(I8), ), I64))
_ECHO_I64 = DeclareFunction('echo_i64', FnType((I64, ), I64))
_ECHO_I32 = DeclareFunction('echo_i32', FnType((I32, ), I64))
_ECHO_PTR = DeclareFunction('echo_ptr', FnType((PointerType(I8), ), I64))
_ECHO_F64 = DeclareFunction('echo_f64', FnType((F64, ), I64))


_echo_buf = ''
@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_char_p)
def _echo_str(ptr):
    global _echo_buf
    _echo_buf += ctypes.string_at(ptr).decode()
    if _echo_buf.endswith('\n'):
        print(_echo_buf.rstrip())
        _echo_buf = ''
    return 0

@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int64)
def _echo_i64(value):
    global _echo_buf
    _echo_buf += str(value)
    if _echo_buf.endswith('\n'):
        print(_echo_buf.rstrip())
        _echo_buf = ''
    return 0

@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int32)
def _echo_i32(value):
    global _echo_buf
    _echo_buf += str(value)
    if _echo_buf.endswith('\n'):
        print(_echo_buf.rstrip())
        _echo_buf = ''
    return 0

@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_double)
def _echo_f64(value):
    global _echo_buf
    _echo_buf += str(value)
    if _echo_buf.endswith('\n'):
        print(_echo_buf.rstrip())
        _echo_buf = ''
    return 0

@ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p)
def _echo_ptr(ptr):
    global _echo_buf
    _echo_buf += f'{ptr}'
    if _echo_buf.endswith('\n'):
        print(_echo_buf.rstrip())
        _echo_buf = ''
    return 0

_GLOBAL_HELPERS = (
    (_ECHO_STR.name, ctypes.cast(_echo_str, ctypes.c_void_p)),
    (_ECHO_I64.name, ctypes.cast(_echo_i64, ctypes.c_void_p)),
    (_ECHO_PTR.name, ctypes.cast(_echo_ptr, ctypes.c_void_p)),
    (_ECHO_I32.name, ctypes.cast(_echo_i32, ctypes.c_void_p)),
    (_ECHO_F64.name, ctypes.cast(_echo_f64, ctypes.c_void_p)),
)
