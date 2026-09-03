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


# -- runtime ``if`` ----------------------------------------------------------


def sign(n):
    # a runtime if whose branches both return
    if n > 0:
        return 1
    else:
        return -1


def clamped(n):
    # the then-branch returns; the else falls through to the code after
    # the if (the trailing return)
    if n > 100:
        return 100
    return n


def classify(n):
    # two consecutive partially-returning runtime ifs
    if n > 100:
        return 1
    if n > 10:
        return 2
    return 3


def nested_if(n):
    # a runtime if whose branches contain further runtime ifs
    if n > 0:
        if n > 100:
            return 2
        else:
            return 1
    else:
        return 0


def bad_join(n):
    # both branches fall through (a join): not supported yet
    if n > 0:
        pass
    else:
        pass
    return n


# -- recursion ---------------------------------------------------------------


def fact(n) -> spy.i32:
    # direct recursion; the return annotation fixes the return type
    if n <= 1:
        return 1
    return n * fact(n - 1)


def fact_aot(n: spy.i32) -> spy.i32:
    if n <= 1:
        return 1
    return n * fact_aot(n - 1)


def gcd[T](a: T, b: T) -> T:
    # recursion whose return annotation is the type parameter ``T``
    if b == 0:
        return a
    return gcd(b, a % b)


def pow2[T](n: T) -> T:
    # a generic recursion that is specialized to several types
    if n <= 0:
        return 1
    return 2 * pow2(n - 1)


def fib(n) -> spy.i32:
    # recursion inside the branches of a runtime if (and a recursive
    # call used twice on one path)
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


def is_even(n) -> spy.bool:
    # mutual recursion with is_odd
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n) -> spy.bool:
    if n == 0:
        return False
    return is_even(n - 1)


def fact_untyped(n):
    # recursion needs the return type while the body is being typed
    if n <= 1:
        return 1
    return n * fact_untyped(n - 1)


def v_fn(a) -> spy.i32:
    return a + 1


def abort_after_call(n):
    # compiles v_fn first (nested, MIR-cached by the module build), then
    # fails on its own unannotated recursion - the whole build aborts
    v_fn(n)
    return abort_after_call(n - 1)


def _make_twin():
    """A factory whose results all share one ``__name__``; registered
    into the same JitContext they must still get distinct native
    symbols (see ``test_same_name_functions_get_distinct_symbols``)."""

    def twin(a, b):
        return a + b

    return twin


_ORIGINALS = {
    name: globals()[name] for name in (
        'add', 'add_aot', 'add_default', 'add_inline', 'foo',
        'is_i32', 'call_add', 'use_default',
        'sign', 'clamped', 'classify', 'nested_if', 'bad_join',
        'fact', 'fact_aot', 'gcd', 'pow2', 'fib', 'is_even', 'is_odd',
        'fact_untyped', 'v_fn', 'abort_after_call',
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
        self.sign = self.cache.jit()(sign)
        self.clamped = self.cache.jit()(clamped)
        self.classify = self.cache.jit()(classify)
        self.nested_if = self.cache.jit()(nested_if)
        self.bad_join = self.cache.jit()(bad_join)
        self.fact = self.cache.jit()(fact)
        self.fact_aot = self.cache.aot()(fact_aot)
        self.gcd = self.cache.jit()(gcd)
        self.pow2 = self.cache.jit()(pow2)
        self.fib = self.cache.jit()(fib)
        self.is_even = self.cache.jit()(is_even)
        self.is_odd = self.cache.jit()(is_odd)
        self.fact_untyped = self.cache.jit()(fact_untyped)
        self.v_fn = self.cache.jit()(v_fn)
        self.abort_after_call = self.cache.jit()(abort_after_call)
        g['add'] = self.add
        g['add_aot'] = self.add_aot
        g['add_default'] = self.add_default
        g['foo'] = self.foo
        g['is_i32'] = self.is_i32
        g['call_add'] = self.call_add
        g['use_default'] = self.use_default
        g['sign'] = self.sign
        g['clamped'] = self.clamped
        g['classify'] = self.classify
        g['nested_if'] = self.nested_if
        g['bad_join'] = self.bad_join
        g['fact'] = self.fact
        g['fact_aot'] = self.fact_aot
        g['gcd'] = self.gcd
        g['pow2'] = self.pow2
        g['fib'] = self.fib
        g['is_even'] = self.is_even
        g['is_odd'] = self.is_odd
        g['fact_untyped'] = self.fact_untyped
        g['v_fn'] = self.v_fn
        g['abort_after_call'] = self.abort_after_call
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

    def test_same_name_functions_get_distinct_symbols(self) -> None:
        # two different functions that share a ``__name__`` must not
        # collide in the context-wide native symbol table (calls between
        # spy functions are linked by symbol name)
        cache = spy.JitContext()
        first = cache.jit()(_make_twin())
        second = cache.jit()(_make_twin())
        self.assertEqual(first(1, 2), 3)
        self.assertEqual(second(1, 2), 3)
        self.assertEqual(first(3, 4), 7)
        self.assertEqual(second(5, 6), 11)
        names = sorted(
            spec.name
            for entry in (first._spy_entry, second._spy_entry)
            for spec in entry.specs.values()
        )
        self.assertEqual(len(names), 2, names)
        # the first registration keeps the plain name, the second one
        # gets a distinct one instead of overwriting it
        self.assertIn('spy.twin.i32.i32', names)
        self.assertEqual(len(set(names)), 2, names)

    # -- runtime ``if`` ------------------------------------------------------

    def test_runtime_if_both_branches_return(self) -> None:
        self.assertEqual(self.sign(5), 1)
        self.assertEqual(self.sign(-3), -1)

    def test_runtime_if_partial_return_falls_through(self) -> None:
        self.assertEqual(self.clamped(50), 50)
        self.assertEqual(self.clamped(500), 100)
        self.assertEqual(self.clamped(-1), -1)

    def test_runtime_if_sequence(self) -> None:
        self.assertEqual(self.classify(1), 3)
        self.assertEqual(self.classify(50), 2)
        self.assertEqual(self.classify(500), 1)

    def test_runtime_if_nested(self) -> None:
        self.assertEqual(self.nested_if(-1), 0)
        self.assertEqual(self.nested_if(5), 1)
        self.assertEqual(self.nested_if(500), 2)

    def test_runtime_if_join_error(self) -> None:
        # a runtime if whose branches both fall through needs a join,
        # which the structured MIR does not support yet
        with self.assertRaises(spy.CompileError) as ctx:
            self.bad_join(1)
        self.assertIn('both fall through', str(ctx.exception))

    # -- recursion -----------------------------------------------------------

    def test_recursion_factorial(self) -> None:
        self.assertEqual(self.fact(1), 1)
        self.assertEqual(self.fact(5), 120)
        self.assertEqual(self.fact(10), 3628800)
        # the second call reuses the compiled specialization
        self.assertEqual(self.fact(10), 3628800)

    def test_recursion_is_a_native_recursive_call(self) -> None:
        self.assertEqual(self.fact(3), 6)
        lines = self._spec_lines(self.fact, (spy.i32,))
        self.assertTrue(
            any('call' in line and 'spy.fact.i32' in line for line in lines), lines
        )

    def test_recursion_aot(self) -> None:
        self.assertEqual(self.fact_aot(5), 120)
        self.assertEqual(self.fact_aot(10), 3628800)

    def test_recursion_generic_over_types(self) -> None:
        # one generic recursion specialized to two types; each
        # specialization recurses to itself
        self.assertEqual(self.pow2(5), 32)
        self.assertEqual(self.pow2(3.0), 8.0)
        names = sorted(spec.name for spec in self.pow2._spy_entry.specs.values())
        self.assertEqual(names, ['spy.pow2.f64', 'spy.pow2.i32'], names)

    def test_recursion_with_type_parameter_return(self) -> None:
        # the return annotation names the type parameter ``T``
        self.assertEqual(self.gcd(48, 36), 12)
        self.assertEqual(self.gcd(17, 5), 1)
        self.assertEqual(self.gcd(2**31 - 1, 3), 1)

    def test_recursion_inside_if_branches(self) -> None:
        # the recursive calls sit in the branch of a runtime if, and one
        # path calls the function twice
        self.assertEqual(self.fib(0), 0)
        self.assertEqual(self.fib(1), 1)
        self.assertEqual(self.fib(10), 55)

    def test_mutual_recursion(self) -> None:
        self.assertTrue(self.is_even(10))
        self.assertFalse(self.is_even(7))
        self.assertTrue(self.is_odd(7))
        self.assertFalse(self.is_odd(10))

    def test_recursion_requires_return_annotation(self) -> None:
        with self.assertRaises(spy.CompileError) as ctx:
            self.fact_untyped(5)
        self.assertIn('requires a return type annotation', str(ctx.exception))
        # compilation failures are deterministic (and cached)
        with self.assertRaises(spy.CompileError):
            self.fact_untyped(5)

    def test_aborted_build_reuses_cached_callees(self) -> None:
        # the failing build of abort_after_call compiled v_fn into its
        # module first (the MIR survives, the spec is never registered);
        # a later direct call of v_fn must still compile and run
        with self.assertRaises(spy.CompileError):
            self.abort_after_call(3)
        self.assertEqual(self.v_fn(41), 42)
        self.assertEqual(self.v_fn(1), 2)


all_tests = [SpyExampleTest]
