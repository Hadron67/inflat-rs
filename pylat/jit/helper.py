from dataclasses import dataclass
from typing import TypeAlias

from .argpass import ComplexType, IntegerType, RealType, TypesConfig, LowerType, ComplexFloatType
from .llvm import BasicBlock, FloatType, FloatValue, IntType, Value
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

    def type_to_lower_type(self, type: ap.Type) -> LowerType:
        match type:
            case ap.IntegerType():
                return self.parent.index_type
            case ap.RealType():
                return self.parent.real_type
            case ap.ComplexType():
                return ComplexFloatType(self.parent.real_type)
            case _:
                raise TypeError(f"cannot convert to lower type: {type}")

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

    def coerce_to_complex_type(self, block: BasicBlock, value: MaybeComplexValue, value_type: ap.Type) -> MaybeComplexValue:
        match value_type:
            case ComplexType():
                return value
            case RealType():
                assert not isinstance(value, ComplexValue)
                return ComplexValue(value, FloatValue(0, self.llvm_real_type))
            case IntegerType():
                assert not isinstance(value, ComplexValue)
                return ComplexValue(block.int_to_float(value, self.llvm_real_type), FloatValue(0, self.llvm_real_type))
            case _:
                raise TypeError(f"cannot coerce type {value_type} to complex")

    def coerce_to_real_type(self, block: BasicBlock, value: Value, value_type: ap.Type):
        match value_type:
            case RealType():
                return value
            case IntegerType():
                return block.int_to_float(value, self.llvm_real_type)
            case _:
                raise TypeError(f"cannot coerce type {value_type} to real")

    def coerce(self, block: BasicBlock, value: MaybeComplexValue, value_type: ap.Type, target_type: ap.Type) -> MaybeComplexValue:
        match target_type:
            case ComplexType():
                return self.coerce_to_complex_type(block, value, value_type)
            case RealType():
                assert not isinstance(value, ComplexValue)
                return self.coerce_to_real_type(block, value, value_type)
            case IntegerType():
                if value_type != IntegerType():
                    raise TypeError(f"cannot coerce {value_type} to integer")
                return value
            case _:
                raise TypeError("????")

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
