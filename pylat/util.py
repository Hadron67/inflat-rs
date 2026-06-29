from inspect import isclass
from typing import get_args, get_origin
from weakref import WeakKeyDictionary

from ast import AST, Assign, Call, Attribute, Constant, Expr, For, FunctionDef, List, Name, Load, Return, Store, Subscript, arguments, expr, stmt, arg

def next_unique_name(prefix: str, used_names: set[str]) -> str:
    i = 0
    while True:
        name = prefix + str(i)
        if name not in used_names:
            used_names.add(name)
            return name
        i += 1

class ObjectCounter[T]:
    _id: WeakKeyDictionary[T, int]
    _next_id: int

    def __init__(self, next_id: int = 0) -> None:
        self._id = WeakKeyDictionary()
        self._next_id = next_id

    def get_id(self, obj: T) -> int:
        if obj in self._id:
            return self._id[obj]
        ret = self._next_id
        self._next_id += 1
        self._id[obj] = ret
        return ret

class StrBiMap[V]:
    _k2v: dict[str, V]
    _v2k: dict[V, str]

    def __init__(self) -> None:
        self._k2v = {}
        self._v2k = {}

    def get_key(self, value: V) -> str:
        return self._v2k[value]

    def get_value(self, key: str) -> V:
        return self._k2v[key]

    def has_key(self, key: str):
        return key in self._k2v

    def has_value(self, value: V):
        return value in self._v2k

    def next_unique_name(self, prefix: str='') -> str:
        if prefix not in self._k2v:
            return prefix
        i = 0
        while True:
            name = prefix + str(i)
            if name not in self._k2v:
                return name
            i += 1

    def add(self, key: str, value: V):
        assert key not in self._k2v
        assert value not in self._v2k
        self._k2v[key] = value
        self._v2k[value] = key

    def items(self):
        return self._k2v.items()

    def keys(self):
        return self._k2v.keys()

    def values(self):
        return self._k2v.values()

class SourceBuilder:
    lines: list[str]
    _indents: int

    def __init__(self, indents: int = 0) -> None:
        self.lines = []
        self._indents = indents

    def emit(self, line: str):
        self.lines.append(('    ' * self._indents) + line)

class SubExprFnBuilder:
    _base_cls: type
    _var_counter: int

    def __init__(self, base_cls: type) -> None:
        self._base_cls = base_cls
        self._var_counter = 0

    def _var(self) -> str:
        ret = f"var{self._var_counter}"
        self._var_counter += 1
        return ret

    def _check_type(self, type: type):
        return isclass(type) and issubclass(type, self._base_cls)

    def _process_one(self, var_name: str, type: type) -> list[str]:
        if self._check_type(type):
            return [f'ret.append({var_name})']
        head = get_origin(type)
        if head is list:
            args = get_args(type)
            v = self._var()
            body = self._process_one(v, args[0])
            if len(body) == 0:
                return []
            return [f"for {v} in {var_name}:", *('    ' + line for line in body)]
        if head is tuple:
            args = get_args(type)
            if len(args) == 2 and args[1] == Ellipsis and self._check_type(args[0]):
                v = self._var()
                body = self._process_one(v, args[0])
                if len(body) == 0:
                    return []
                return [f"for {v} in {var_name}:", *('    ' + line for line in body)]
            else:
                lines: list[str] = []
                for i, arg in enumerate(args):
                    lines.extend(self._process_one(f"{var_name}[{i}]", arg))
                return lines
        return []

    def generate_get_children(self, cls: type, fn_name: str, excludes: set[str] | None = None):
        body: list[str] = [f'def {fn_name}(self):', "    ret = []"]
        for b in cls.mro()[-1::-1]:
            if b is object or b is self._base_cls:
                continue
            for name, type in b.__annotations__.items():
                if excludes is not None and name in excludes:
                    continue
                body.extend(('    ' + i) for i in self._process_one(f"self.{name}", type))
        body.append('    return ret')
        return body

def resize[T](arr: list[T], value: T, length: int):
    while len(arr) < length:
        arr.append(value)

def add_line_numbers(lines: list[str]):
    max_num_len = len(str(len(lines) + 1))
    ret: list[str] = []
    for i, line in enumerate(lines):
        n = str(i + 1)
        ret.append(n + (' ' * (max_num_len - len(n))) + '|' + line)
    return ret
