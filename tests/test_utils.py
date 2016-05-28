# -*- coding: utf-8 -*-
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

    def test_mul_ad_ad(self):
        # Answers computed using MRST's initVariablesADI
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4,5]]).T)
        z = x*y
        assert np.array_equal(z.val, np.array([[4, 10]]).T)
        assert np.array_equal(np.array([[4,0],[0,5]]), z.jac[0].toarray())
        assert np.array_equal(np.array([[1,0],[0,2]]), z.jac[1].toarray())

        f = x*x*y
        g = x*y*z
        h = f*g
        assert np.array_equal(h.val, np.array([[64, 2000]]).T)
        assert np.array_equal(np.array([[256,0],[0,4000]]), h.jac[0].toarray())
        assert np.array_equal(np.array([[48,0],[0,1200]]), h.jac[1].toarray())

        w = f*g + f + g*x*y
        assert np.array_equal(w.val, np.array([[132, 3020]]).T)
        assert np.array_equal(np.array([[456,0],[0,5520]]), w.jac[0].toarray())
        assert np.array_equal(np.array([[97,0],[0,1804]]), w.jac[1].toarray())

    def test_mul_ad_scalar(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[4,5,6]]).T)
        w = x*x*y*3 + x*y*y*5
        assert np.array_equal(np.array([[92, 310, 702]]).T, w.val)
        assert np.array_equal(np.array([[104,0,0],[0,185,0], [0,0,288]]), w.jac[0].toarray())
        assert np.array_equal(np.array([[43,0,0],[0,112,0], [0,0,207]]), w.jac[1].toarray())

    def test_mul_ad_vector(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[4,5,6]]).T)
        w = x*x*y*np.array([[3],[3],[3]]) + x*y*y*np.array([[5],[5],[5]])
        assert np.array_equal(np.array([[92, 310, 702]]).T, w.val)
        assert np.array_equal(np.array([[104,0,0],[0,185,0], [0,0,288]]), w.jac[0].toarray())
        assert np.array_equal(np.array([[43,0,0],[0,112,0], [0,0,207]]), w.jac[1].toarray())

    def test_mul_ADI3_ADI1(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4]]).T)
        s1 = y*x
        s2 = x*y
        assert np.array_equal(np.array([[4, 8]]).T, s1.val)
        assert np.array_equal(np.array([[4,0],[0,4]]), s1.jac[0].toarray())
        assert np.array_equal(np.array([[1],[2]]), s1.jac[1].toarray())
        assert np.array_equal(np.array([[4, 8]]).T, s2.val)
        assert np.array_equal(np.array([[4,0],[0,4]]), s2.jac[0].toarray())
        assert np.array_equal(np.array([[1],[2]]), s2.jac[1].toarray())

    def test_rmul(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[4,5,6]]).T)
        z = 3*x
        assert np.array_equal(np.array([[3, 6, 9]]).T, z.val)
        assert np.array_equal(np.array([[3,0,0],[0,3,0], [0,0,3]]), z.jac[0].toarray())
        assert np.array_equal(np.array([[0,0,0],[0,0,0], [0,0,0]]), z.jac[1].toarray())
        w = 3*x*x*y + 5*x*y*y
        assert np.array_equal(np.array([[92, 310, 702]]).T, w.val)
        assert np.array_equal(np.array([[104,0,0],[0,185,0], [0,0,288]]), w.jac[0].toarray())
        assert np.array_equal(np.array([[43,0,0],[0,112,0], [0,0,207]]), w.jac[1].toarray())

    def test_pow_ad_scalar_and_ad_vec_len1(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4,5]]).T)
        z = x**2
        assert np.array_equal(np.array([[1, 4]]).T, z.val)
        assert np.array_equal(np.array([[2,0],[0,4]]), z.jac[0].toarray())
        assert np.array_equal(np.array([[0,0],[0,0]]), z.jac[1].toarray())

    def test_pow_ad_ad_samelen(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[4,5]]).T)
        z = x**y
        dz2dy2 = 2**5 * np.log(2)
        print(z.jac[1].toarray())
        assert np.array_equal(np.array([[1, 32]]).T, z.val)
        assert np.array_equal(np.array([[4,0],[0,80]]), z.jac[0].toarray())
        assert np.array_equal(np.array([[0,0],[0,dz2dy2]]), z.jac[1].toarray())

    def test_pow_ad_scalar_or_scalar_ad(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[2,3,4]]).T)
        u = x+y
        r = 2**u
        s = u**2

        drdx = 2**u.val * np.log(2) * u.jac[0].toarray()
        drdy = 2**u.val * np.log(2) * u.jac[1].toarray()
        assert np.array_equal(np.array([[8, 32, 128]]).T, r.val)
        assert np.array_equal(drdx, r.jac[0].toarray())
        assert np.array_equal(drdy, r.jac[1].toarray())

        assert np.array_equal(np.array([[9, 25, 49]]).T, s.val)
        assert np.array_equal(np.array([[6,0,0],[0,10,0],[0,0,14]]), s.jac[0].toarray())
        assert np.array_equal(np.array([[6,0,0],[0,10,0],[0,0,14]]), s.jac[1].toarray())

    def test_pow_different_len(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[2,3,4,5]]).T)
        with pytest.raises(ValueError):
            x**y
        with pytest.raises(ValueError):
            x**y.val
        with pytest.raises(ValueError):
            x.val**y

    def test_div(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[2,3,4]]).T)
        u = x+2*y

        # ADI divided by scalar
        s = u/2
        assert np.array_equal(np.array([[2.5, 4, 5.5]]).T, s.val)
        assert np.array_equal(np.array([[0.5,0,0],[0,0.5,0],[0,0,0.5]]), s.jac[0].toarray())
        assert np.array_equal(np.array([[1,0,0],[0,1,0],[0,0,1]]), s.jac[1].toarray())

        # Scalar divided by ADI
        s = 440/u
        assert np.array_equal(np.array([[88, 55, 40]]).T, s.val)
        assert np.allclose(np.array([[-17.6,0,0],[0,-6.875,0],[0,0,-3.636363636363637]]), s.jac[0].toarray())
        assert np.allclose(np.array([[-35.2,0,0],[0,-13.75,0],[0,0,-7.272727272727273]]), s.jac[1].toarray())

        # ADI divided by ADI
        s = u / (x+y)
        assert np.allclose(np.array([[1.666666666666667, 1.6, 1.571428571428571]]).T, s.val)
        assert np.allclose(np.array([[-0.222222222222222,0,0],[0,-0.12,0],[0,0,-0.081632653061225]]), s.jac[0].toarray())
        assert np.allclose(np.array([[0.111111111111111,0,0],[0,0.08,0],[0,0,0.061224489795918]]), s.jac[1].toarray())

        # ADI divided by ADI of length 1
        x, y = initVariablesADI(np.array([[8,2]]).T, np.array([[2]]).T)
        s = x/y
        assert np.array_equal(np.array([[4,1]]).T, s.val)
        assert np.array_equal(np.array([[0.5,0],[0,0.5]]), s.jac[0].toarray())
        assert np.array_equal(np.array([[-2],[-0.5]]), s.jac[1].toarray())

        # ADI of length 1 divided by vector
        s = y/np.array([[4],[2]])
        assert isinstance(s, ADI)
        assert isinstance(s.jac[0], sps.spmatrix)
        assert isinstance(s.jac[1], sps.spmatrix)
        assert np.array_equal(np.array([[0.5,1]]).T, s.val)
        assert np.array_equal(np.array([[0,0],[0,0]]), s.jac[0].toarray())
        assert np.array_equal(np.array([[0.25],[0.5]]), s.jac[1].toarray())

    def test_getitem(self):
        x, y, z = initVariablesADI(np.array([[0,1,2,3]]).T, np.array([[0,1,2]]).T, np.array([[1]]))
        x0 = x[0]
        assert np.array_equal(np.array([[0]]).T, x0.val)
        assert np.array_equal(np.array([[1,0,0,0]]), x0.jac[0].toarray())
        assert np.array_equal(np.array([[0,0,0]]), x0.jac[1].toarray())
        assert np.array_equal(np.array([[0]]), x0.jac[2].toarray())

        x0 = x[(2,1),:]
        assert np.array_equal(np.array([[2,1]]).T, x0.val)
        assert np.array_equal(np.array([[0,0,1,0],
                                        [0,1,0,0]]), x0.jac[0].toarray())
        assert np.array_equal(np.array([[0,0,0],
                                        [0,0,0]]), x0.jac[1].toarray())
        assert np.array_equal(np.array([[0],[0]]), x0.jac[2].toarray())

    def test_setitem(self):
        x, y = initVariablesADI(np.array([[0,1]]).T, np.array([[5]]).T)
        x[0] = x[1]
        assert x.val[0,0] == x.val[1,0]
        assert np.array_equal(x[0].jac[0].toarray(), x[1].jac[0].toarray())
        x[0] = 99
        assert x.val[0,0] == 99
        assert np.array_equal(x.jac[0].toarray(), np.array([[0,0],[0,1]]))

    def test_dot(self):
        # TODO complete test
        pass
