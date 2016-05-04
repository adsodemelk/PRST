from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import scipy.sparse as sps

from prst.utils import rlencode, rldecode

from prst.utils import mcolon, initVariablesADI, ADI
from prst.utils.units import *


class Test_rlencode:
    """See also doctests in utils.py"""
    def test_example(self):
        A = np.array([
            [1,2,3,4],
            [1,2,3,4],
            [3,4,5,6],
            [3,3,3,3],
            [3,3,4,5],
            [3,3,4,5]])
        A_rl, n = rlencode(A, 0)
        assert np.array_equal(A_rl.shape, np.array([4, 4]))
        assert n.size == 4
        assert np.array_equal(A_rl, np.array([
            [1,2,3,4],
            [3,4,5,6],
            [3,3,3,3],
            [3,3,4,5]]))
        assert np.array_equal(n, np.array([2,1,1,2]))


    def test_example_transposed(self):
        A = np.array([
            [1,1,3,3,3,3],
            [2,2,4,3,3,3],
            [3,3,5,3,4,4],
            [4,4,6,3,5,5]])
        A_rl, n = rlencode(A, 1)
        assert np.array_equal(A_rl.shape, np.array([4, 4]))
        assert n.size == 4
        assert np.array_equal(A_rl, np.array([
            [1,3,3,3],
            [2,4,3,3],
            [3,5,3,4],
            [4,6,3,5]]))
        assert np.array_equal(n, np.array([2,1,1,2]))

    def test_1d_array(self):
        A = np.array([1,1,1,1,4,4,5])
        A_rl, n = rlencode(A)
        assert np.array_equal(A_rl, np.array([1,4,5]))
        assert np.array_equal(n, np.array([4,2,1]))


class Test_rldecode:
    """See also doctests in utils.py"""
    def test_1d_A_2d_n(self):
        A = np.array([7,8,9])
        n = np.array([[2],[3], [0]])
        B1 = rldecode(A, n, axis=0)
        assert np.array_equal(B1, np.array([7, 7, 8, 8, 8]))
        # A only has one axis, so axis=1 does not make sense
        with pytest.raises(ValueError):
            B2 = rldecode(A, n, axis=1)


class Test_units:
    def test_basic_units(self):
        assert centi == 1/100

class Test_mcolon:
    def test_basic(self):
        lo = np.array([1, 2])
        hi = np.array([3, 4])
        ans = np.array([1, 2, 2, 3])
        assert np.array_equal(mcolon(lo, hi), ans)

    def test_stride(self):
        lo = np.array([1, 2])
        hi = np.array([6, 14])
        s = np.array([2, 3])
        ans = np.array([1,3,5, 2,5,8,11])
        assert np.array_equal(mcolon(lo, hi, s), ans)

class Test_ADI:
    def test_init(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[4,5]]).T)
        assert np.array_equal(x.val, np.array([[1,2,3]]).T)
        assert np.array_equal(y.val, np.array([[4,5]]).T)
        assert len(x.jac) == 2
        assert len(y.jac) == 2
        assert (x.jac[0] - sps.eye(3)).nnz == 0
        assert (x.jac[1] - sps.csr_matrix((3,2))).nnz == 0
        assert (y.jac[0] - sps.csr_matrix((2,3))).nnz == 0
        assert (y.jac[1] - sps.eye(2)).nnz == 0

    def test_copy(self):
        # Simply assigning returns a reference
        x, = initVariablesADI(np.array([[1,2,3]]).T)
        y = x
        x.val[0,:] = 10
        assert y.val[0,:] == 10
        # Need copy to actually make a copy
        z = x.copy()
        z.val[0,:] = 11
        assert x.val[0,:] == 10

    def test_len(self):
        x, = initVariablesADI(np.array([[1,2,3,4]]).T)
        assert len(x) == 4

    def test_shape(self):
        x, = initVariablesADI(np.array([[1,2,3,4]]).T)
        assert x.shape[0] == 4
        assert x.shape[1] == 1

    def test_dim(self):
        x, = initVariablesADI(np.array([[1,2,3]]).T)
        assert x.ndim == 2

    def test_ge_gt_le_lt(self):
        x_val = np.array([[1,2,3,2,4]]).T
        y_val = np.array([[4,3,2,2,1]]).T
        x, y = initVariablesADI(x_val, y_val)
        assert np.array_equal( x_val >= y_val, x >= y )
        assert np.array_equal( x_val > y_val, x > y )
        assert np.array_equal( x_val <= y_val, x <= y )
        assert np.array_equal( x_val < y_val, x < y )

    def test_pos(self):
        x_val = np.array([[1,2,3,2,4]]).T
        x, = initVariablesADI(x_val)
        y = +x
        assert np.array_equal(y.val, x.val)
        assert all([(y_jac-x_jac).nnz == 0 for (y_jac, x_jac) in zip(y.jac, x.jac)])

    def test_neg(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[4,5]]).T)
        z = -x
        assert np.array_equal(z.val, -x.val)
        assert (x.jac[0] + z.jac[0]).nnz == 0

    def test_add(self):
        # 1) Both AD, same length
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4,5]]).T)
        z = x + y
        assert np.array_equal(z.val, x.val + y.val)
        assert all([(jz - jx - jy).nnz == 0 for (jx, jy, jz) in zip(x.jac, y.jac, z.jac)])

        # 2) Both AD, len(v)==1
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        z = ADI(np.array([[1]]), [sps.csr_matrix(np.array([[2,2]])), sps.csr_matrix(np.array([[1]]))])
        w = x + z
        assert np.array_equal(w.val, x.val + z.val)
        assert np.array_equal(w.val, np.array([[2,3]]).T)
        assert np.array_equal(w.jac[0].toarray(), np.eye(2) + np.array([[2,2],[2,2]]))
        assert np.array_equal(w.jac[1].toarray(), np.array([[1], [1]]))

        # 3) Both AD, len(u)==1
        w = z + x
        assert np.array_equal(w.val, x.val + z.val)
        assert np.array_equal(w.val, np.array([[2,3]]).T)
        assert np.array_equal(w.jac[0].toarray(), np.eye(2) + np.array([[2,2],[2,2]]))
        assert np.array_equal(w.jac[1].toarray(), np.array([[1], [1]]))

        # 4) u AD, v scalar, len(v)==1
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        w = x + 5
        assert np.array_equal(w.val, np.array([[1+5,2+5]]).T)
        assert np.array_equal(w.jac[0].toarray(), x.jac[0].toarray())
        assert np.array_equal(w.jac[1].toarray(), x.jac[1].toarray())
        w2 = 5 + x
        assert np.array_equal(w2.val, np.array([[1+5,2+5]]).T)
        assert np.array_equal(w.jac[0].toarray(), x.jac[0].toarray())
        assert np.array_equal(w.jac[1].toarray(), x.jac[1].toarray())

        # 5) u AD, v vector, same length
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        w1 = x + np.array([[2,1]]).T
        assert np.array_equal(w1.val, np.array([[1+2,2+1]]).T)
        assert np.array_equal(w1.jac[0].toarray(), x.jac[0].toarray())
        assert np.array_equal(w1.jac[1].toarray(), x.jac[1].toarray())
        w2 = np.array([[2,1]]).T + x
        assert np.array_equal(w2.val, np.array([[1+2,2+1]]).T)
        assert np.array_equal(w2.jac[0].toarray(), x.jac[0].toarray())
        assert np.array_equal(w2.jac[1].toarray(), x.jac[1].toarray())

        # 6) Different length AD vectors
        with pytest.raises(ValueError):
            x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[1,2,3]]).T)
            x + y

    def test_sub(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        z = x - y
        assert np.array_equal(z.val, x.val - y.val)

    def test_rsub(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        z = 5 - x
        assert np.array_equal(z.val, 5-x.val)
        assert (x+z).jac[0].nnz == 0
        assert (x+z).jac[1].nnz == 0

    def test_mul(self):
        # z = x*y = [1*4, 2*5]' = [4, 10]
        # dz/dx = x*Jy + y*Jx = x + y   // identity Jacobians
        #       = [1]*[1 0]  +  [4][1 0]
        #         [2]*[0 1]  +  [5][0 1]
        # = [1 0] + [4 0] = [5 0]
        #   [0 2] + [0 5] = [0 7]
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4,5]]).T)
        z = x*y
        assert np.array_equal(z.val, np.array([[4, 10]]).T)
        assert np.array_equal(np.array([[5,0],[0,7]]), z.jac[0].toarray())

    def test_dot(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        y.dot(5)
        # TODO complete test
