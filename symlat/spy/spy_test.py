"""Integration tests for the spy JIT (``symlat.spy``).

The functions under test are plain module-level Python functions (they
are the *source* spy compiles, obtained with ``inspect.getsource``).
Every test registers them into a fresh :class:`JitContext` and rebinds
the module-level names to the returned wrappers before the first call,
so that function bodies referring to other spy functions (like ``foo``
calling ``add_aot``/``add_inline``) resolve to the wrappers of that
context.
"""

# pyright: reportGeneralTypeIssues=false
# pyright: reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false
# pyright: reportReturnType=false
# The sample functions are DSL source compiled by spy, not ordinary
# Python that a type checker could understand (e.g. ``a + b`` with ``a,
# b: T``).

import io
from contextlib import redirect_stdout
from unittest import TestCase

from .. import spy

# ---------------------------------------------------------------------------
# functions under test (the example of spy/instructions.md)
# ---------------------------------------------------------------------------


def add[T](a: T, b: T) -> T:
    return a + b


def add_aot(a: spy.u64, b: spy.u64) -> spy.u64:
    return a + b


def add_default[T](a: T, b: T = 0) -> T:
    return a + b


def add_inline[T](a: T, b: T) -> T:
    spy.compile_log("add_inline was compiled")
    return a + b


def foo(a, b):
    if spy.type(a) == spy.u64 and spy.type(b) == spy.u64:
        return add_aot(a, b)
    else:
        return add_inline(a, b)


def is_i32(a):
    return spy.type(a) == spy.i32


def bad_aot(a: spy.u64, b: spy.u64) -> spy.u64:
    # body computes a float; the return type annotation does not match
    return a + b + 1.5


def call_add[T](a: T, b: T) -> T:
    # calling another jit function with fresh types triggers a nested
    # compilation during the compile of call_add
    return add(a, b)


def use_default[T](a: T) -> T:
    # call a spy function whose default parameter applies
    return add_default(a)


_ORIGINALS = {
    name: globals()[name] for name in (
        'add', 'add_aot', 'add_default', 'add_inline', 'foo',
        'is_i32', 'call_add', 'use_default',
    )
}


class SpyExampleTest(TestCase):
    def setUp(self) -> None:
        self.cache = spy.JitContext()
        g = globals()
        self.add = self.cache.jit()(add)
        self.add_aot = self.cache.aot()(add_aot)
        self.add_default = self.cache.jit()(add_default)
        self.foo = self.cache.jit()(foo)
        self.is_i32 = self.cache.jit()(is_i32)
        self.call_add = self.cache.jit()(call_add)
        self.use_default = self.cache.jit()(use_default)
        g['add'] = self.add
        g['add_aot'] = self.add_aot
        g['add_default'] = self.add_default
        g['foo'] = self.foo
        g['is_i32'] = self.is_i32
        g['call_add'] = self.call_add
        g['use_default'] = self.use_default
        # add_inline stays the raw function: it is *inlined*, not compiled
        self.addCleanup(self._restore_globals)

    def _restore_globals(self) -> None:
        globals().update(_ORIGINALS)

    def _spec_lines(self, wrapper, arg_types) -> list[str]:
        entry = wrapper._spy_entry
        spec = entry.specs[arg_types]
        return spec.lines

    # -- the example of instructions.md --------------------------------------

    def test_jit_int(self) -> None:
        self.assertEqual(self.add(1, 2), 3)
        self.assertEqual(self.add(7, 8), 15)

    def test_jit_float(self) -> None:
        self.assertEqual(self.add(1.0, 2.0), 3.0)
        self.assertEqual(self.add(0.5, 0.25), 0.75)

    def test_jit_strings_fail_to_compile(self) -> None:
        with self.assertRaises(spy.CompileError) as ctx:
            self.add('', '')
        message = str(ctx.exception)
        self.assertIn("'+'", message)
        self.assertIn('strings are compiled as arrays of u8', message)
        # compilation failure is deterministic (and cached)
        with self.assertRaises(spy.CompileError):
            self.add('x', 'y')

    def test_jit_conflicting_types(self) -> None:
        with self.assertRaises(spy.TypeMismatchError) as ctx:
            self.add(1, 2.0)
        self.assertIn('conflicting types', str(ctx.exception))

    def test_aot(self) -> None:
        self.assertEqual(self.add_aot(1, 2), 3)
        # large unsigned values round-trip
        self.assertEqual(self.add_aot(2**63 - 1, 2), 2**63 + 1)

    def test_aot_type_mismatch(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self.add_aot(1.0, 2.0)
        self.assertIn('type mismatch', str(ctx.exception))

    def test_aot_out_of_range(self) -> None:
        with self.assertRaises(spy.TypeMismatchError) as ctx:
            self.add_aot(-1, 2)
        self.assertIn('out of range', str(ctx.exception))

    def test_aot_as_mismatch(self) -> None:
        with self.assertRaises(spy.TypeMismatchError):
            self.add_aot(spy.as_(1, spy.i32), spy.as_(2, spy.u64))

    def test_aot_return_type_mismatch_at_decoration(self) -> None:
        cache = spy.JitContext()
        with self.assertRaises(spy.CompileError):
            cache.aot()(bad_aot)

    def test_defaults(self) -> None:
        self.assertEqual(self.add_default(1, 2), 3)
        self.assertEqual(self.add_default(1), 1)
        # keyword arguments may come in any order
        self.assertEqual(self.add_default(b=45, a=12), 57)
        self.assertEqual(self.add_default(a=3, b=4), 7)
        # default arguments are marshaled to the resolved type parameter
        self.assertEqual(self.add_default(spy.as_(1, spy.u64)), 1)

    def test_missing_argument(self) -> None:
        with self.assertRaises(TypeError):
            self.add_default()

    def test_unknown_kwarg(self) -> None:
        with self.assertRaises(TypeError):
            self.add_default(a=1, c=2)

    def test_foo_u64_branch_calls_add_aot(self) -> None:
        # the compile-time branch picks add_aot: no log, and the compiled
        # function contains a native call to the aot specialization
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = self.foo(spy.as_(1, spy.u64), spy.as_(2, spy.u64))
        self.assertEqual(result, 3)
        self.assertEqual(buf.getvalue(), '')
        lines = self._spec_lines(self.foo, (spy.u64, spy.u64))
        self.assertTrue(any('call' in line and 'spy.add_aot.u64.u64' in line for line in lines))

    def test_foo_else_branch_inlines(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(self.foo(1, 2), 3)
            # the specialization is cached: no second compile log
            self.assertEqual(self.foo(3, 4), 7)
        self.assertEqual(buf.getvalue(), 'add_inline was compiled\n')
        lines = self._spec_lines(self.foo, (spy.i32, spy.i32))
        self.assertFalse(any('spy.add_aot' in line for line in lines))

    def test_comptime_type_query(self) -> None:
        self.assertTrue(self.is_i32(1))
        self.assertFalse(self.is_i32(1.0))
        self.assertTrue(self.is_i32(spy.as_(3, spy.i32)))

    def test_nested_jit_compilation(self) -> None:
        # call_add compiles a fresh specialization of add on the fly
        self.assertEqual(self.call_add(2, 3), 5)
        # add's i32 specialization now exists and is reused by add itself
        self.assertEqual(self.add(10, 20), 30)

    def test_defaults_inside_function_body(self) -> None:
        self.assertEqual(self.use_default(41), 41)
        self.assertEqual(self.use_default(1.5), 1.5)


all_tests = [SpyExampleTest]
