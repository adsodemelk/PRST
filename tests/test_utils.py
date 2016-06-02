# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import scipy.sparse as sps

from prst.utils import rlencode, rldecode, recursive_diff, npad

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

# Only to satisfy pytest-cov. Not actually testing anything.
class Test_recursive_diff:
    recursive_diff({'a':5}, {'b': 6})
    recursive_diff(np.array([0.00000001]), np.array([0.0000]))

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

    def test_jacnotlist(self):
        x = ADI(np.array([[1,2,3]]).T, sps.eye(3))
        assert (x.jac[0] - sps.eye(3)).nnz == 0

    def test_repr(self):
        x, y = initVariablesADI(np.array([[1,2,3]]).T, np.array([[4,5]]).T)
        y.__repr__()


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

    """
    def test_len(self):
        x, = initVariablesADI(np.array([[1,2,3,4]]).T)
        with pytest.raises(NotImplementedError):
            len(x)
    """

    def test_shape(self):
        x, = initVariablesADI(np.array([[1,2,3,4]]).T)
        assert x.shape[0] == 4
        assert x.shape[1] == 1

    def test_dim(self):
        x, = initVariablesADI(np.array([[1,2,3]]).T)
        assert x.ndim == 2

    def test_ge_gt_le_lt(self):
        x_val = np.array([[1,2,3,2,4,6,7]]).T
        y_val = np.array([[4,3,2,2,8,9,10]]).T
        x, y = initVariablesADI(x_val, y_val)
        assert np.array_equal( x_val >= y_val, x >= y )
        assert np.array_equal( x_val >= 5, x >= 5 )

        assert np.array_equal( x_val > y_val, x > y )
        assert np.array_equal( x_val > 5, x > 5 )

        assert np.array_equal( x_val <= y_val, x <= y )
        assert np.array_equal( x_val <= 5, x <= 5 )

        assert np.array_equal( x_val < y_val, x < y )
        assert np.array_equal( x_val < 5, x < 5 )

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

    def test_mul_ad_ad_mismatch(self):
        a, b = initVariablesADI(np.array([[1,2]]).T, np.array([[1,2,3]]).T)
        with pytest.raises(ValueError):
            a*b

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
        z = np.array([[2]])
        f = x*z
        assert np.array_equal(np.array([[2, 4, 6]]).T, f.val)
        assert np.array_equal(np.array([[2,0,0],[0,2,0], [0,0,2]]), f.jac[0].toarray())
        assert np.array_equal(np.array([[0,0,0],[0,0,0], [0,0,0]]), f.jac[1].toarray())
        with pytest.raises(ValueError):
            x*np.array([[1,2]]).T

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

    def test_truediv(self):
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

    def test_div(self):
        u, = initVariablesADI(np.array([[1]]))
        # uses "classic" division even if future division is enabled.
        import operator
        with pytest.raises(DeprecationWarning):
            operator.div(u, 5)
        with pytest.raises(DeprecationWarning):
            operator.div(5, u)

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
        with pytest.raises(ValueError):
            y[np.array([[1, 1], [2,2]])]

    def test_setitem(self):
        x, y = initVariablesADI(np.array([[0,1]]).T, np.array([[5]]).T)
        x[0] = x[1]
        assert x.val[0,0] == x.val[1,0]
        assert np.array_equal(x[0].jac[0].toarray(), x[1].jac[0].toarray())
        x[0] = 99
        assert x.val[0,0] == 99
        assert np.array_equal(x.jac[0].toarray(), np.array([[0,0],[0,1]]))

    def test_max(self):
        x, y = initVariablesADI(np.array([[0,1]]).T, np.array([[5]]).T)
        xmax = x.max()
        assert xmax.val[0,0] == 1
        assert xmax.ndim == 2
        assert np.array_equal(xmax.jac[0].toarray(), np.array([[0,1]]))
        assert np.array_equal(xmax.jac[1].toarray(), np.array([[0]]))

    def test_min(self):
        x, y = initVariablesADI(np.array([[1,2]]).T, np.array([[5]]).T)
        xmin = x.min()
        assert xmin.val[0,0] == 1
        assert xmin.ndim == 2
        assert np.array_equal(xmin.jac[0].toarray(), np.array([[1,0]]))
        assert np.array_equal(xmin.jac[1].toarray(), np.array([[0]]))

    def test_sum(self):
        x, y = initVariablesADI(np.array([[0,1]]).T, np.array([[5]]).T)
        z = x+y # [5;6]
        assert z.val[0,0] == 5 and z.val[1,0] == 6
        assert np.array_equal(z.jac[0].toarray(), np.array([[1,0],[0,1]]))
        assert np.array_equal(z.jac[1].toarray(), np.array([[1],[1]]))
        sumz = z.sum()
        assert sumz.val[0,0] == 11
        assert np.array_equal(sumz.jac[0].toarray(), np.array([[1,1]]))
        assert np.array_equal(sumz.jac[1].toarray(), np.array([[2]]))

    def test_sin(self):
        x, y = initVariablesADI(np.array([[0,np.pi/2]]).T, np.array([[5]]).T)
        sinx = x.sin()
        assert isinstance(sinx, ADI)
        assert abs(sinx.val[0,0]) < 0.0001
        assert abs(sinx.val[1,0]-1) < 0.0001
        assert abs(sinx.jac[0][0,0] - 1) < 0.0001
        assert abs(sinx.jac[0][1,1]) < 0.0001

    def test_cos(self):
        x, y = initVariablesADI(np.array([[0,np.pi/2]]).T, np.array([[5]]).T)
        cosx = x.cos()
        assert isinstance(cosx, ADI)
        assert abs(cosx.val[0,0] - 1) < 0.0001
        assert abs(cosx.val[1,0]) < 0.0001
        assert abs(cosx.jac[0][0,0]) < 0.0001
        assert abs(cosx.jac[0][1,1] + 1) < 0.0001

    def test_npcos(self):
        x, y = initVariablesADI(np.array([[0,np.pi/2]]).T, np.array([[5]]).T)
        cosx = np.cos(x)
        assert isinstance(cosx, ADI)
        assert np.allclose(cosx.val, np.array([[1, 0]]).T)
        assert np.allclose(cosx.jac[0].toarray(), np.array([[0,0],[0,-1]]))
        assert np.allclose(cosx.jac[1].toarray(), np.array([[0],[0]]))

    def test_npmultiply(self):
        x, y = initVariablesADI(np.array([[4,2]]).T, np.array([[2,3]]).T)
        z = np.multiply(x,y)
        assert z.val[0,0] == 8 and z.val[1,0] == 6
        assert z.jac[0][0,0] == 2 and z.jac[0][1,1] == 3
        assert z.jac[1][0,0] == 4 and z.jac[1][1,1] == 2

    def test_dot_mat_ad(self):
        x, y = initVariablesADI(np.array([[4,2]]).T, np.array([[2,3]]).T)
        z = x*y #[8; 6] with jac[0] = diag(2,3), jac[1] = diag(4,2)
        A = np.array([[1, 2], [3, 4]])
        w = npad.dot(A, z)
        assert isinstance(w, ADI)
        assert np.array_equal(w.val, np.array([[20, 48]]).T)
        assert np.array_equal(w.jac[0].toarray(), np.array([[2, 6], [6, 12]]))
        assert np.array_equal(w.jac[1].toarray(), np.array([[4, 4], [12, 8]]))

    def test_dot_ad_ad(self):
        x, y = initVariablesADI(np.array([[4]]).T, np.array([[2]]).T)
        z = x+y
        w = npad.dot(z, x)
        assert isinstance(w, ADI)
        assert np.array_equal(w.val, np.array([[24]]).T)
        assert np.array_equal(w.jac[0].toarray(), np.array([[10]]))
        assert np.array_equal(w.jac[1].toarray(), np.array([[4]]))

    def test_dot_mat_vec(self):
        x, y = np.array([[4,2]]).T, np.array([[2,3]])
        w = npad.dot(x, y)
        assert isinstance(w, np.ndarray)
        assert np.array_equal(w, np.array([[8, 12], [4, 6]]))

    def test_dot_chain(self):
        x, y = initVariablesADI(np.array([[4,2]]).T, np.array([[2,3]]).T)
        z = x.dot(9)
        assert isinstance(z, ADI)
        assert np.array_equal(z.val, np.array([[36, 18]]).T)
        z = x.dot(3).dot(3)
        assert isinstance(z, ADI)
        assert np.array_equal(z.val, np.array([[36, 18]]).T)

    def test_exp(self):
        x, y = initVariablesADI(np.array([[4,2]]).T, np.array([[2,3]]).T)
        z = x.exp()
        assert np.allclose(z.val, np.array([[54.5982, 7.3891]]).T)
        assert np.allclose(z.jac[0].toarray(), np.array([[54.5982, 0], [0, 7.3891]]))
        assert np.allclose(z.jac[1].toarray(), np.array([[0, 0], [0, 0]]))
        w = np.exp(x)
        assert np.allclose(w.val, np.array([[54.5982, 7.3891]]).T)

    def test_log(self):
        x, y = initVariablesADI(np.array([[4,2]]).T, np.array([[2,3]]).T)
        z = x.log()
        assert np.allclose(z.val, np.array([[1.386294361119891, 0.693147]]).T)
        assert np.allclose(z.jac[0].toarray(), np.array([[0.2500, 0], [0, 0.5000]]))
        assert np.allclose(z.jac[1].toarray(), np.array([[0,0], [0,0]]))
        w = np.log(x)
        assert np.allclose(w.val, np.array([[1.3863, 0.693147]]).T)

    def test_sign(self):
        x, y = initVariablesADI(np.array([[4,2]]).T, np.array([[2,3,-5]]).T)
        ysign = y.sign()
        assert np.array_equal(ysign, np.array([[1,1,-1]]).T)
        wsign = npad.sign(y)
        assert np.array_equal(wsign, np.array([[1,1,-1]]).T)

    def test_abs(self):
        x, y = initVariablesADI(np.array([[5, -2]]).T, np.array([[3]]).T)
        z1 = (x*y).abs()
        assert np.array_equal(z1.val, np.array([[15, 6]]).T)
        assert (z1.jac[0] - sps.diags([3, -3], 0)).nnz == 0
        assert np.array_equal(z1.jac[1].toarray(), np.array([[5], [2]]))
        z2 = npad.abs(x*y)
        assert np.array_equal(z2.val, np.array([[15, 6]]).T)
        assert (z2.jac[0] - sps.diags([3, -3], 0)).nnz == 0
        assert np.array_equal(z2.jac[1].toarray(), np.array([[5], [2]]))

    def test_tile(self):
        x, y = initVariablesADI(np.array([[5, -2]]).T, np.array([[3]]).T)
        yyy = npad.tile(y, (3,1))
        assert np.array_equal(yyy.val, np.array([[3,3,3]]).T)
        assert np.array_equal(yyy.jac[1].toarray(), np.array([[1,1,1]]).T)
        with pytest.raises(TypeError):
            npad.tile(y, 3)
        with pytest.raises(TypeError):
            npad.tile(y, (3,2))

    def test_vstack(self):
        x, y = initVariablesADI(np.array([[5, -2]]).T, np.array([[3]]).T)
        xy = npad.vstack((x,y))
        assert np.array_equal(xy.val, np.array([[5, -2, 3]]).T)
        assert np.array_equal(xy.jac[0].toarray(), np.array([[1,0], [0,1], [0,0]]))
        assert np.array_equal(xy.jac[1].toarray(), np.array([[0],[0],[1]]))

    def test_concatenate(self):
        x, y = initVariablesADI(np.array([[5, -2]]).T, np.array([[3]]).T)
        xy = npad.concatenate((x,y), axis=0)
        assert np.array_equal(xy.val, np.array([[5, -2, 3]]).T)
        with pytest.raises(TypeError):
            npad.concatenate((x,y), axis=1)

    def test_numpy_ufunc(self):
        x, = initVariablesADI(np.array([[5]]))
        with pytest.raises(NotImplementedError):
            print(x.__numpy_ufunc__("func", "method", "pos", "inputs"))

class Test_npad_non_ad:
    def test_vec_vec(self):
        a, b = np.array([[1,-2]]), np.array([[3,4]])
        assert np.array_equal(npad.tile(a, (3,2)), np.tile(a, (3,2)))
        assert np.array_equal(npad.sign(a), np.sign(a))
        assert np.array_equal(npad.abs(a), np.abs(a))
