"""User-facing JIT wrapper that compiles element-wise array functions.

The ``Wrapper`` class turns a plain Python function whose body consists of
element-wise array assignments into a JIT-compiled kernel that runs in-place on
numpy arrays.  The assignments are parsed from the function source with the
``ast`` module and translated into :class:`pylat.expr.AssignExpr`, which is then
compiled with :class:`pylat.jit.compile.JitCompiler` on the first call.  The
result is cached per (dtype, rank) signature of the arguments.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping

import numpy as np
from llvmlite import binding as llvm

from ..expr import (
    AssignExpr,
    Cos,
    Exp,
    Expr,
    Int,
    Ln,
    Power,
    Rational,
    Sin,
    Times,
    symbol,
)
from .argpass import ComplexFloatType, FloatType, IntType, LowerType, TypeContext
from .backend import Backend
from .compile import CompiledWrapper, JitCompiler
from .openmp import OpenMPBackend

# augmented assignment operators supported by _compile_assignment
_OP_MAP = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
}

_FUNC_MAP = {
    'sin': Sin,
    'cos': Cos,
    'exp': Exp,
    'ln': Ln,
    'log': Ln,
}


class _ExprTranslator:
    """Translates a Python expression AST into a pylat ``Expr``."""

    def __init__(self, names: Mapping[str, Expr]) -> None:
        self._names = names
        self.used_names: set[str] = set()

    def _call_name(self, func: ast.AST) -> str:
        match func:
            case ast.Name(id):
                return id
            case ast.Attribute(ast.Name(id), attr):
                return f'{id}.{attr}'
        raise TypeError(f"unsupported callable {type(func).__name__}")

    def translate(self, node: ast.AST) -> Expr:
        match node:
            case ast.Name(id):
                if id not in self._names:
                    raise TypeError(f"undefined name {id!r}; only function parameters can be referenced")
                self.used_names.add(id)
                return self._names[id]
            case ast.Constant(value):
                if isinstance(value, bool):
                    raise TypeError(f"unsupported constant {value!r}")
                if isinstance(value, (int, float, complex)):
                    return Expr.as_expr(value)
                raise TypeError(f"unsupported constant {value!r}")
            case ast.BinOp(left, op, right):
                lhs = self.translate(left)
                rhs = self.translate(right)
                match op:
                    case ast.Add():
                        return lhs + rhs
                    case ast.Sub():
                        return lhs - rhs
                    case ast.Mult():
                        return lhs * rhs
                    case ast.Div():
                        return lhs / rhs
                    case ast.Pow():
                        return lhs ** rhs
                raise TypeError(f"unsupported binary operator {type(op).__name__}")
            case ast.UnaryOp(op, operand):
                value = self.translate(operand)
                match op:
                    case ast.USub():
                        return Times((Int(-1), value))
                    case ast.UAdd():
                        return value
                raise TypeError(f"unsupported unary operator {type(op).__name__}")
            case ast.Call(func, args, keywords):
                if len(args) != 1 or len(keywords) > 0:
                    raise TypeError("only single-argument function calls are supported")
                name = self._call_name(func).split('.')[-1]
                if name == 'sqrt':
                    return Power(self.translate(args[0]), Rational(1, 2))
                if name in _FUNC_MAP:
                    return _FUNC_MAP[name](self.translate(args[0]))
                raise TypeError(f"unsupported function call {name!r}")
            case _:
                raise TypeError(f"unsupported expression node {type(node).__name__}")


def _parse_fn(fn: Callable) -> tuple[tuple[str, ...], set[str], list[AssignExpr]]:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise TypeError(
            f"cannot inspect the source of {fn!r}; the function must be defined in a module or a file"
        ) from e
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise TypeError(f"expected a function definition, got {type(tree.body[0]).__name__}")
    node = tree.body[0]
    params = tuple(a.arg for a in node.args.args)
    if len(set(params)) != len(params):
        raise TypeError("duplicate parameter names are not allowed")
    names = {name: symbol(name) for name in params}
    translator = _ExprTranslator(names)

    assigns: list[AssignExpr] = []
    for stmt in node.body:
        # skip the docstring
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise TypeError("assignment target must be a single parameter name")
            lhs = stmt.targets[0].id
            op = ''
        elif isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise TypeError("augmented assignment target must be a parameter name")
            lhs = stmt.target.id
            op = _OP_MAP.get(type(stmt.op))
            if op is None:
                raise TypeError(f"unsupported augmented assignment operator {type(stmt.op).__name__}")
        else:
            raise TypeError(f"unsupported statement {type(stmt).__name__}; only assignments are allowed")
        if lhs not in names:
            raise TypeError(f"unknown parameter {lhs!r}")
        translator.used_names.add(lhs)
        rhs = translator.translate(stmt.value)
        assigns.append(AssignExpr(names[lhs], rhs, op))

    if len(assigns) == 0:
        raise TypeError("the function body contains no assignments")
    return params, translator.used_names, assigns


class _JittedFunction:
    """The callable produced by ``Wrapper.jit``."""

    def __init__(self, wrapper: 'Wrapper', fn: Callable, assigns: list[AssignExpr], params: tuple[str, ...], used_names: set[str]) -> None:
        self._wrapper = wrapper
        self._assigns = assigns
        self._params = params
        self._used_names = used_names
        self._names = {name: symbol(name) for name in params}
        self._cache: dict[tuple[tuple[LowerType, int], ...], CompiledWrapper] = {}
        self.__name__ = getattr(fn, '__name__', 'jitted')
        self.__doc__ = getattr(fn, '__doc__', None)

    def _infer_arg_type(self, value) -> tuple[LowerType, int]:
        if isinstance(value, np.ndarray):
            return LowerType.from_numpy_dtype(str(value.dtype)), value.ndim
        if isinstance(value, (np.floating, float)):
            return self._wrapper._real_type, 0
        if isinstance(value, (np.complexfloating, complex)):
            return ComplexFloatType(self._wrapper._real_type), 0
        if isinstance(value, (np.integer, int)):
            return self._wrapper._index_type, 0
        raise TypeError(f"unsupported argument type: {type(value).__name__}")

    def _compile(self, signature: tuple[tuple[LowerType, int], ...]) -> CompiledWrapper:
        context = TypeContext()
        for name, (lower_type, dim) in zip(self._params, signature):
            context.set_symbol(self._names[name], lower_type, dim)
        compiler = JitCompiler(
            self._wrapper._backend,
            real_type=self._wrapper._real_type,
            index_type=self._wrapper._index_type,
        )
        return compiler.compile_assignments(self._assigns, context)

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise TypeError(f"{self.__name__}() does not support keyword arguments")
        if len(args) != len(self._params):
            raise TypeError(
                f"{self.__name__}() takes {len(self._params)} positional arguments but {len(args)} were given"
            )
        signature = tuple(self._infer_arg_type(arg) for arg in args)
        compiled = self._cache.get(signature)
        if compiled is None:
            compiled = self._compile(signature)
            self._cache[signature] = compiled
        arg_map = {self._names[n]: v for n, v in zip(self._params, args) if n in self._used_names}
        return compiled.call(arg_map)

    def print_all(self):
        """Print the LLVM IR of the most recently compiled kernel."""
        if len(self._cache) == 0:
            return []
        return list(self._cache.values())[-1].print_all()

    def __repr__(self) -> str:
        return f"<jitted {self.__name__}>"


class Wrapper:
    """
    Usage example:

    ```python
    wrapper = Wrapper()

    @wrapper.jit()
    def my_func(a, b, c, dt):
        a += c * dt
        b += c * dt + c * 2

    a = np.random.rand(8, 9, 10)
    b = np.random.rand(8, 9, 10)
    c = np.random.rand(8, 9, 10)
    a0 = a.copy()
    b0 = b.copy()
    dt = 0.5
    my_func(a, b, c, dt)
    assert np.allclose(a, a0 + c * dt)
    assert np.allclose(b, b0 + c * dt + c * 2)
    ```
    """

    def __init__(self, backend: Backend | None = None, real_type: FloatType | None = None, index_type: IntType | None = None) -> None:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self._backend = backend if backend is not None else OpenMPBackend()
        self._real_type = real_type if real_type is not None else FloatType(64)
        self._index_type = index_type if index_type is not None else IntType(64, False)

    def jit(self, fn: Callable | None = None):
        """Decorator that compiles a function of element-wise array assignments."""
        def decorator(f: Callable) -> _JittedFunction:
            params, used_names, assigns = _parse_fn(f)
            return _JittedFunction(self, f, assigns, params, used_names)
        if fn is not None:
            return decorator(fn)
        return decorator
