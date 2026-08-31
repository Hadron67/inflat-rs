import ctypes
from unittest import TestCase

import numpy as np
from llvmlite import binding as llvm
from numpy.testing import assert_almost_equal

from ..expr import AssignExpr, Int, Plus, Rational, S, Slice, Times, symbols
from .compile import CompiledWrapper, JitCompiler, StandardLayoutMode
from .fn_wrapper import Wrapper
from .openmp import OpenMPBackend
from .type import ComplexFloatType, FloatType, SymbolTypeDesc

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class TestExpr(TestCase):
    def __init__(self, methodName: str = "test_evaluation") -> None:
        super().__init__(methodName)

    def test_evaluation(self):
        x, y = symbols('x', 'y')
        self.assertEqual(
            (x + x * 2 + S(2) * y + y * 3 + y / 2).normalize(),
            Plus((Times((Int(3), x)), Times((Rational(11, 2), y))))
        )

class JitTest(TestCase):
    def __init__(self, methodName: str = "test_jit") -> None:
        super().__init__(methodName)

    def test_assignment(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([
            (phi, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (mom_phi, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (dt, SymbolTypeDesc(FloatType(64), 0)),
        ], [
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ])

        np.random.seed(114514)
        phi0 = np.zeros((10, 10, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(10, 10, 10) + np.random.randn(10, 10, 10) * 1j
        dt0 = 2.0

        fn.call(phi0, mom_phi0, dt0)

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

    def test_assignment_non_uniform_shape(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([
            (phi, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (mom_phi, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (dt, SymbolTypeDesc(FloatType(64), 0)),
        ], [
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ])

        np.random.seed(114514)
        phi0 = np.zeros((8, 9, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(8, 9, 10) + np.random.randn(8, 9, 10) * 1j
        dt0 = 2.0

        fn.call(phi0, mom_phi0, dt0)

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)

    def test_sum(self):
        a, = symbols('a')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([(a, SymbolTypeDesc(FloatType(64), 3))], a)

        np.random.seed(114514)
        a0 = np.random.randn(10, 10, 10)
        result = fn.call(a0)
        assert_almost_equal(result, np.sum(a0))

    def test_sum_complex(self):
        a, b = symbols('a', 'b')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([
            (a, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (b, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
        ], a * b + a)

        np.random.seed(42)
        a0 = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        b0 = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        result = fn.call(a0, b0)
        assert_almost_equal(result, np.sum(a0 * b0 + a0))

    def test_sum_with_assignment(self):
        phi, mom_phi, dt = symbols('phi', 'mom_phi', 'dt')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([
            (phi, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (mom_phi, SymbolTypeDesc(ComplexFloatType(FloatType(64)), 3)),
            (dt, SymbolTypeDesc(FloatType(64), 0)),
        ], [
            AssignExpr(phi, mom_phi * mom_phi * dt)
        ], reduction=mom_phi)

        np.random.seed(114514)
        phi0 = np.zeros((8, 9, 10), dtype=np.complex128)
        mom_phi0 = np.random.randn(8, 9, 10) + np.random.randn(8, 9, 10) * 1j
        dt0 = 2.0

        result = fn.call(phi0, mom_phi0, dt0)

        assert_almost_equal(phi0, mom_phi0 * mom_phi0 * dt0)
        assert_almost_equal(result, np.sum(mom_phi0))

    def test_multiple_assignments_same_shape(self):
        # regression: several assignments accumulating into the same array must
        # resolve the shared shape constraints transitively instead of failing
        a, b, c, d = symbols('a', 'b', 'c', 'd')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments([
            (a, SymbolTypeDesc(FloatType(64), 1)),
            (b, SymbolTypeDesc(FloatType(64), 1)),
            (c, SymbolTypeDesc(FloatType(64), 1)),
            (d, SymbolTypeDesc(FloatType(64), 1)),
        ], [
            AssignExpr(a, b, '+'),
            AssignExpr(a, c, '+'),
            AssignExpr(a, d, '+'),
        ])

        np.random.seed(40)
        a0 = np.zeros(5)
        b0 = np.random.rand(5)
        c0 = np.random.rand(5)
        d0 = np.random.rand(5)
        fn.call(a0, b0, c0, d0)
        assert_almost_equal(a0, b0 + c0 + d0)

    def test_scalar_ref(self):
        x, y = symbols('x', 'y')

        compiler = JitCompiler(OpenMPBackend())
        # written references are compiled as pointers and writes propagate back
        fn = compiler.compile_assignments([
            (x, SymbolTypeDesc(FloatType(64), 0, is_ref=True)),
            (y, SymbolTypeDesc(FloatType(64), 0, is_ref=True)),
        ], [
            AssignExpr(x, x, '+'),
            AssignExpr(y, x, '+'),
        ])

        x0 = ctypes.c_double(2.0)
        y0 = ctypes.c_double(3.0)
        fn.call(x0, y0)
        self.assertEqual(x0.value, 4.0)
        self.assertEqual(y0.value, 7.0)

        # 0-d numpy arrays are addressable too
        x2 = np.array(2.0)
        y2 = ctypes.c_double(3.0)
        fn.call(x2, y2)
        self.assertEqual(x2[()], 4.0)
        self.assertEqual(y2.value, 7.0)

        # references that are only read are demoted to by-value scalars
        a, = symbols('a')
        fn2 = compiler.compile_assignments([
            (a, SymbolTypeDesc(FloatType(64), 1)),
            (x, SymbolTypeDesc(FloatType(64), 0, is_ref=True)),
        ], [
            AssignExpr(a, x, '+'),
        ])
        a1 = np.zeros(5)
        x1 = ctypes.c_double(3.0)
        fn2.call(a1, x1)
        assert_almost_equal(a1, np.full(5, 3.0))
        self.assertEqual(x1.value, 3.0)
class SimdLayoutTests(TestCase):
    """Tests for the SIMD friendly linear-index kernels.

    When the expressions contain no rolls and no interior slices, and every array
    argument is a standard layout (C or F contiguous), the kernel skips the
    unpack/repack of the loop index and accesses arrays directly with the flat
    loop variable.
    """

    @staticmethod
    def _last_compiled(f) -> CompiledWrapper:
        # the cached value is (main_kernel, *sum_kernels), converter
        return list(f._cache.values())[-1][0][0]

    def test_row_major_uses_flat_kernel(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c, dt):
            a += b * c + b * dt

        np.random.seed(3)
        a = np.random.rand(8, 9, 10)
        b = np.random.rand(8, 9, 10)
        c = np.random.rand(8, 9, 10)
        a0, b0, c0 = a.copy(), b.copy(), c.copy()
        f(a, b, c, 0.5)
        assert_almost_equal(a, a0 + b0 * c0 + b0 * 0.5)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)
        # the flat kernel must not unpack the loop index (no unsigned div/rem)
        ir = '\n'.join(self._last_compiled(f).print_all())
        self.assertNotIn('urem', ir)
        self.assertNotIn('udiv', ir)

    def test_column_major_uses_flat_kernel(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += b * c + b

        np.random.seed(4)
        a = np.asfortranarray(np.random.rand(6, 7, 8))
        b = np.asfortranarray(np.random.rand(6, 7, 8))
        c = np.asfortranarray(np.random.rand(6, 7, 8))
        a0, b0, c0 = a.copy(), b.copy(), c.copy()
        f(a, b, c)
        assert_almost_equal(a, a0 + b0 * c0 + b0)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.COLUMN_MAJOR)
        ir = '\n'.join(self._last_compiled(f).print_all())
        self.assertNotIn('urem', ir)
        self.assertNotIn('udiv', ir)

    def test_non_contiguous_arrays_fall_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b * 2

        np.random.seed(5)
        a = np.zeros((4, 5))
        b = np.random.rand(5, 4)
        a0 = a.copy()
        f(a, b.T)  # transposed views are not standard layout
        assert_almost_equal(a, a0 + b.T * 2)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_mixed_layout_falls_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += b * c

        np.random.seed(6)
        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        c = np.asfortranarray(np.random.rand(4, 5))
        a0 = a.copy()
        f(a, b, c)
        assert_almost_equal(a, a0 + b * c)
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_axis0_slice_uses_flat_kernel(self):
        # slicing the max-stride axis is linear, including negative indices
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1]
            a += b[0]
            a += b[-1]

        a = np.zeros(6)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, b[1] + b[0] + b[-1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

    def test_middle_slice_falls_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[:, 1]

        a = np.zeros(5)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, b[:, 1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_multi_axis_slices(self):
        # a slice fixing the leading axes is an affine offset of the flat index
        # in row-major layout, so it stays in the standard layout kernel
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1, 2]

        a = np.zeros(7)
        b = np.arange(3 * 4 * 7).reshape(3, 4, 7).astype(float)
        f(a, b)
        assert_almost_equal(a, b[1, 2])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

        # fixing an interior axis (b[1, :, 2]) is not an affine offset of the
        # flat index, so the generic kernel is used
        @wrapper.jit()
        def g(a, b):
            a += b[1, :, 2]

        a = np.zeros(4)
        b = np.arange(3 * 4 * 7).reshape(3, 4, 7).astype(float)
        g(a, b)
        assert_almost_equal(a, b[1, :, 2])
        self.assertIs(self._last_compiled(g).standard_layout, StandardLayoutMode.NONE)

    def test_roll_falls_back_to_generic(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.roll(b, 1, axis=0)

        a = np.zeros((5, 6))
        b = np.random.rand(5, 6)
        f(a, b)
        assert_almost_equal(a, np.roll(b, 1, axis=0))
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.NONE)

    def test_1d_lhs_slice_uses_flat_kernel(self):
        # the sliced base has a higher rank than the loop; the flat kernel handles it
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b[1] + b[-1]

        a = np.zeros(6)
        b = np.arange(30).reshape(5, 6).astype(float)
        f(a, b)
        assert_almost_equal(a, b[1] + b[-1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

    def test_slice_of_compound_expression_uses_flat_kernel(self):
        # slicing a function application (or any compound expression over a single
        # array) is linear: np.sin(b)[2][i] == np.sin(b[2][i])
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sin(b)[2]
            a += np.cos(b)[-1]

        np.random.seed(11)
        a = np.zeros(6)
        b = np.random.rand(5, 6) + 1
        f(a, b)
        assert_almost_equal(a, np.sin(b)[2] + np.cos(b)[-1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

    def test_slice_of_compound_expression_no_unpack(self):
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += np.sin(b)[2]

        np.random.seed(12)
        a = np.zeros(6)
        b = np.random.rand(5, 6) + 1
        f(a, b)
        assert_almost_equal(a, np.sin(b)[2])
        ir = '\n'.join(self._last_compiled(f).print_all())
        self.assertNotIn('urem', ir)

    def test_compound_expression_slice_uses_flat_kernel(self):
        # a slice over an expression with several distinct arrays is linear too: each
        # base array gets the fixed index applied with its own stride
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b, c):
            a += (b + c)[1]

        np.random.seed(13)
        a = np.zeros(6)
        b = np.random.rand(5, 6)
        c = np.random.rand(5, 6)
        f(a, b, c)
        assert_almost_equal(a, (b + c)[1])
        self.assertIs(self._last_compiled(f).standard_layout, StandardLayoutMode.ROW_MAJOR)

        # the same expression with non-contiguous arrays must stay correct via the
        # generic kernel
        @wrapper.jit()
        def g(a, b):
            a += np.sin(b)[2]

        a = np.zeros(6)
        b = np.random.rand(6, 5)
        g(a, b.T)
        assert_almost_equal(a, np.sin(b.T)[2])
        self.assertIs(self._last_compiled(g).standard_layout, StandardLayoutMode.NONE)

    def test_layout_cache_variants(self):
        # C-contiguous and F-contiguous calls compile separate kernels
        wrapper = Wrapper()

        @wrapper.jit()
        def f(a, b):
            a += b * 2

        a = np.zeros((4, 5))
        b = np.random.rand(4, 5)
        f(a, b)
        self.assertEqual(len(f._cache), 1)
        a = np.asfortranarray(np.zeros((4, 5)))
        b = np.asfortranarray(np.random.rand(4, 5))
        f(a, b)
        self.assertEqual(len(f._cache), 2)
        # the cached value is (main_kernel, *sum_kernels), converter
        self.assertEqual(
            {w[0].standard_layout for w, _ in f._cache.values()},
            {StandardLayoutMode.ROW_MAJOR, StandardLayoutMode.COLUMN_MAJOR},
        )

    def test_flat_reduction(self):
        a, = symbols('a')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([(a, SymbolTypeDesc(FloatType(64), 3))], a, standard_layout=StandardLayoutMode.ROW_MAJOR)
        self.assertIs(fn.standard_layout, StandardLayoutMode.ROW_MAJOR)

        np.random.seed(7)
        a0 = np.random.rand(10, 10, 10)
        assert_almost_equal(fn.call(a0), np.sum(a0))

    def test_flat_reduction_column_major(self):
        a, = symbols('a')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([(a, SymbolTypeDesc(FloatType(64), 2))], a, standard_layout=StandardLayoutMode.COLUMN_MAJOR)
        self.assertIs(fn.standard_layout, StandardLayoutMode.COLUMN_MAJOR)

        np.random.seed(8)
        a0 = np.asfortranarray(np.random.rand(10, 10))
        assert_almost_equal(fn.call(a0), np.sum(a0))

    def test_flat_reduction_with_slice(self):
        b, = symbols('b')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_reduction([(b, SymbolTypeDesc(FloatType(64), 2))], Slice(b, ((0, 1),)), standard_layout=StandardLayoutMode.ROW_MAJOR)
        self.assertIs(fn.standard_layout, StandardLayoutMode.ROW_MAJOR)

        np.random.seed(9)
        b0 = np.random.rand(5, 6)
        assert_almost_equal(fn.call(b0), np.sum(b0[1]))

    def test_mismatched_layout_raises(self):
        # the direct JitCompiler API verifies the layout of the arguments at call time
        a, b = symbols('a', 'b')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments(
            [(a, SymbolTypeDesc(FloatType(64), 2)), (b, SymbolTypeDesc(FloatType(64), 2))], [AssignExpr(a, b, '+')],
            standard_layout=StandardLayoutMode.ROW_MAJOR,
        )
        a0 = np.zeros((4, 5))
        b0 = np.asfortranarray(np.random.rand(4, 5))
        with self.assertRaises(ValueError):
            fn.call(a0, b0)

    def test_generic_rank_mismatched_slices(self):
        # regression: the generic kernel must handle sliced arrays whose rank is
        # higher than the loop rank (e.g. a += b[1] with a one-dimensional a)
        a, b = symbols('a', 'b')

        compiler = JitCompiler(OpenMPBackend())
        fn = compiler.compile_assignments(
            [(a, SymbolTypeDesc(FloatType(64), 1)), (b, SymbolTypeDesc(FloatType(64), 2))], [AssignExpr(a, Slice(b, ((0, 1),)), '+')]
        )
        a0 = np.zeros(6)
        b0 = np.arange(30).reshape(5, 6).astype(float)
        fn.call(a0, b0)
        assert_almost_equal(a0, b0[1])

        b3, = symbols('b3')
        fn3 = compiler.compile_assignments(
            [(a, SymbolTypeDesc(FloatType(64), 1)), (b3, SymbolTypeDesc(FloatType(64), 3))],
            [AssignExpr(a, Slice(Slice(b3, ((2, 2),)), ((0, 1),)), '+')],
        )
        a3 = np.zeros(4)
        b30 = np.arange(3 * 4 * 7).reshape(3, 4, 7).astype(float)
        fn3.call(a3, b30)
        assert_almost_equal(a3, b30[1, :, 2])

        # a sliced array that is broadcast into a higher-rank loop (base rank is
        # smaller than the loop rank) must align its surviving axis with the
        # trailing loop axes
        fn4 = compiler.compile_assignments(
            [(a, SymbolTypeDesc(FloatType(64), 3)), (b, SymbolTypeDesc(FloatType(64), 2))], [AssignExpr(a, Slice(b, ((0, 1),)), '+')]
        )
        a4 = np.zeros((2, 3, 6))
        b40 = np.arange(30).reshape(5, 6).astype(float)
        fn4.call(a4, b40)
        assert_almost_equal(a4, np.broadcast_to(b40[1], a4.shape))

        b5, = symbols('b5')
        fn5 = compiler.compile_assignments(
            [(a, SymbolTypeDesc(FloatType(64), 2)), (b5, SymbolTypeDesc(FloatType(64), 4))],
            [AssignExpr(a, Slice(Slice(b5, ((2, 3),)), ((1, 2),)), '+')],
        )
        a5 = np.zeros((2, 4))
        b50 = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(float)
        fn5.call(a5, b50)
        assert_almost_equal(a5, b50[:, 2, 3][:, :4])

all_tests = [TestExpr, JitTest, SimdLayoutTests]
