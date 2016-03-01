from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np

from prst.utils import rlencode, rldecode

from prst.utils import mcolon
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
