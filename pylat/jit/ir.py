from dataclasses import dataclass


class Type:
    pass

@dataclass
class IntegerType(Type):
    pass

class RealType(Type):
    pass

class ComplexType(Type):
    pass
