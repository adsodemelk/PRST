# -*- coding: utf-8 -*-
from __future__ import print_function, division
import copy

__all__ = ["rldecode", "rlencode", "units", "mcolon", "recursive_diff", "gridtools"]

import prst.utils.gridtools

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
import scipy.sparse as sps

class Struct(dict):
    """
    MATLAB-struct-like object.

    Source: http://stackoverflow.com/questions/35988/

    """
    def __init__(self, **kwargs):
        super(Struct, self).__init__(**kwargs)
        self.__dict__ = self


def rlencode(A, axis=0):
    """
    Compute run length encoding of array A along axis.

    Synopsis:
        A, n = rlencode(A)
        A, n = rlencode(A, axis)

    Arguments:
        A (np.ndarray): Array to be encoded.
        axis (Optional[int]): Axis of A where run length encoding is done.
                              Default value: axis=0

    Example (default axis):
        >>> A = np.array([
        ...     [1, 2, 3, 4],
        ...     [1, 2, 3, 4],
        ...     [3, 4, 5, 6],
        ...     [3, 3, 3, 3],
        ...     [3, 3, 4, 5],
        ...     [3, 3, 4, 5]])
        >>> A, n = rlencode(A, 0)
        >>> print(A)
        [[1 2 3 4]
         [3 4 5 6]
         [3 3 3 3]
         [3 3 4 5]]
        >>> print(n)
        [2 1 1 2]

    Example (j-axis):
        >>> A = np.array([
        ...     [1,1,3,3,3,3],
        ...     [2,2,4,3,3,3],
        ...     [3,3,5,3,4,4],
        ...     [4,4,6,3,5,5]])
        >>> A, n = rlencode(A, 1)
        >>> print(A)
        [[1 3 3 3]
         [2 4 3 3]
         [3 5 3 4]
         [4 6 3 5]]
        >>> print(n)
        [2 1 1 2]
    """
    # Let the relevant axis be the first axis
    B = np.swapaxes(A, 0, axis)

    # Flatten axes that are normal to the encoding axis
    B = B.reshape([B.shape[0],-1])

    # Pick indices where the next index is different
    i = np.append(np.where(np.any(B[:-1] != B[1:], axis=1)), B.shape[0]-1)

    # Find the number of repetitions
    n = np.diff(np.insert(i, 0, -1))

    # Pick necessary slices of the encoding axis
    return A.take(i, axis=axis), n


def rldecode(A, n, axis=0):
    """
    Decompresses run length encoding of array A along axis.

    Synopsis:
        B = rldecode(A, n, axis)
        B = rldecode(A, n)        # axis assumed to be 0

    Arguments:
        A (np.ndarray): Encoded array
        n (np.ndarray): Repetition of each layer along an axis.
        axis (Optional[int]): Axis of A where run length decoding is done.

    Returns:
        Uncompressed matrix

    Example (1D-array) along default axis:
        >>> A = np.array([1,4,5])
        >>> n = np.array([4,2,1])
        >>> print(rldecode(A, n))
        [1 1 1 1 4 4 5]

    Example (2D-array) along j-axis:
        >>> A = np.array([
        ...     [1,3,3,3],
        ...     [2,4,3,3],
        ...     [3,5,3,4],
        ...     [4,6,3,5]])
        >>> n = np.array([2,1,1,2])
        >>> print(rldecode(A, n, axis=1))
        [[1 1 3 3 3 3]
         [2 2 4 3 3 3]
         [3 3 5 3 4 4]
         [4 4 6 3 5 5]]
    """
    assert n.size > 0, "Length array was empty."
    # repeat functions take 1d array
    if n.ndim != 1:
        assert n.ndim <= 2
        assert n.shape[0] == 1 or n.shape[1] == 1
        n = n.ravel()
    return A.repeat(n, axis=axis)

def mcolon(lo, hi, s=None):
    """
    Compute concatenated ranges.

    Synopsis:
        mcolon(lo, hi)
        mcolon(lo, hi, stride)

    Arguments:
        lo (ndarray):
            1d array of lower bounds
        hi (ndarray):
            1d array of upper bounds
        s (Optional[ndarray]):
            1d array of strides. Default = np.ones(lo.shape) (unit strides).

    Returns:
        np.r_[lo[0]:hi[0], ..., lo[-1]:hi[-1]]
        np.r_[lo[0]:hi[0]:s[0], ..., lo[-1]:hi[-1]:s[-1]]
        (The NumPy r_ index trick builds a concatenated array of ranges.)

    Example:
        >>> lo = np.array([0,0,0,0])
        >>> hi = np.array([2,3,4,5])
        >>> ind = mcolon(lo, hi)
        >>> np.array_equal(ind, np.array([0,1,0,1,2,0,1,2,3,0,1,2,3,4]))
        True
    """
    if s is None:
        ranges = [range(l,h) for (l,h) in zip(lo,hi)]
    else:
        ranges = [range(l,h,st) for (l,h,st) in zip(lo,hi,s)]
    return np.concatenate(ranges)

def recursive_diff(A, B, indent=0):
    """
    Shows which attributes differ between two objects. Recursive.

    Synopsis:
        recursive_diff(A, B)

    Example:
        >> from prst.gridprocessing import cartGrid
        >> G, V = cartGrid([3,3,3]), cartGrid([3,3,4])
        >> recursive_diff(G, V)
        ====== Recursive comparison ======
         gridType
           Equal, (list,list)
         cells
           facePos
             NOT EQUAL, (ndarray,ndarray)
           num
             NOT EQUAL, (int,int)
           indexMap
             NOT EQUAL, (ndarray,ndarray)
        ...

    """
    def pprint(*args, **kwargs):
        print(" "*indent, *args, **kwargs)

    if indent == 0:
        print()
        print("====== Recursive comparison ======")

    # For classes, try to get dict attribute
    try:
        A = A.__dict__
    except:
        pass
    try:
        B = B.__dict__
    except:
        pass
    if isinstance(A, dict) and isinstance(B, dict):
        # Descend into attributes which exist in both and are dicts. Print them first.
        pass
        inA = set(A.keys())
        inB = set(B.keys())
        notInA = inB - inA
        notInB = inA - inB
        inBoth = inA & inB
        # Print attributes only in A
        if notInA:
            pprint("A MISSING ATTRIBUTES:", notInA)
        # Print attributes only in B
        if notInB:
            pprint("B MISSING ATTRIBUTES:", notInB)
        # Recursively do the same with common attributes
        for attr in inBoth:
            pprint(attr)
            recursive_diff(A[attr], B[attr], indent+2)

    else:
        # Compare A, B for equality
        equal = False
        try:
            equal = None
            close = None
            if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
                equal = np.array_equal(A, B)
                close = np.allclose(A, B)
            else:
                equal = A == B
            if equal:
                pprint("Equal, ", end="")
            else:
                pprint("NOT EQUAL, ", end="")
                if close:
                    print("(BUT APPROXIMATELY EQUAL)", end="")
        except:
            pprint("NOT COMPARABLE, ", end="")

        print("("+A.__class__.__name__+","+B.__class__.__name__+")")


class ADI(object):
    """ADI: Automatic DIfferentiation

    Simple implementation of automatic differentiation for easy construction
    of Jacobian matrices.

    Synopsis:
        x = ADI(value, jacobian)

    Arguments:
        value(np.ndarray):
            The numerical value of the object. Must be a NumPy column array.
            Not compatible with matrices.

        jacobian(list[scipy.sparse.csr_matrix]):
            The Jacobian of the object. Split into parts to improve performance.

    Comment:
        This class is typically instantiated for a set of variables using
        initVariablesADI, not by itself.

    See also:
        initVariablesADI
    """

    def __init__(self, val, jac):
        self.val = val
        self.jac = jac
        self.__array_priority__ = 100
        if not isinstance(self.jac, list):
            self.jac = [self.jac,]

    def __repr__(self):
        jacstring = str([block.shape for block in self.jac])
        return "(val: {0}.T, jac block sizes: {1})".format(self.val.T, jacstring)

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.val)

    @property
    def shape(self):
        return self.val.shape

    @property
    def ndim(self):
        return self.val.ndim

    def __ge__(u, v):
        return u.val >= v.val

    def __gt__(u, v):
        return u.val > v.val

    def __le__(u, v):
        return u.val <= v.val

    def __lt__(u, v):
        return u.val < v.val

    def __pos__(u): # +u
        return u.copy()

    def __neg__(u): # -u
        return ADI(-u.val, [-j for j in u.jac])

    def __add__(u, v): # u + v
        if isinstance(v, ADI):
            if len(u.val) == len(v.val):
                return ADI(u.val + v.val, [ju+jv for (ju,jv) in zip(u.jac, v.jac)])
            if len(v.val) == 1:
                # Tile v.jac to same length as u.jac since sparse matrices
                # don't broadcast properly.
                # https://github.com/scipy/scipy/issues/2128
                vjac = [sps.bmat([[j]]*len(u.val)) for j in v.jac]
                retjac = [ju+jv for (ju,jv) in zip(u.jac, vjac)]
                return ADI(u.val+v.val, retjac)
            if len(u.val) == 1:
                # Vice versa, this time tile u instead
                ujac = [sps.bmat([[j]]*len(v.val)) for j in u.jac]
                retjac = [ju+jv for (ju,jv) in zip(ujac, v.jac)]
                return ADI(u.val+v.val, retjac)
            raise ValueError("Dimension mismatch")
        # v isn't AD object
        v = np.atleast_2d(v)
        return ADI(u.val + v, copy.deepcopy(u.jac))

    def __radd__(v, u): # u + v
        return v.__add__(u)

    def __sub__(u, v):
        return u.__add__(-v)

    def __rsub__(v, u): # u - v
        return (-v).__add__(u)

    # mul
    def __mul__(u, v):
        """Hadamard product u*v."""
        if isinstance(v, ADI):
            if len(u.val) == len(v.val):
                # Note: scipy.sparse.diags has changed parameters between
                # versions 0.16x and 0.17x. This code is only tested on 0.16x.
                # TODO test code in SciPy 0.17x
                uJv = [sps.diags([u.val.flat],[0])*jv for jv in v.jac] # MATRIX multiplication
                vJu = [sps.diags([v.val.flat],[0])*ju for ju in u.jac] # MATRIX multiplication
                jac = [a+b for (a,b) in zip(uJv, vJu)]
                return ADI(u.val*v.val, jac)
            if len(v.val) == 1:
                # Fix dimensions and recurse
                vval = np.tile(v.val, (u.val.shape[0],1) )
                vjac = [sps.bmat([[j]]*len(u.val)) for j in v.jac]
                return u.__mul__(ADI(vval, vjac))
            if len(u.val) == 1:
                # Fix dimensions and recurse
                uval = np.tile(u.val, (v.val.shape[0],1) )
                ujac = [sps.bmat([[j]]*len(v.val)) for j in u.jac]
                return ADI(uval, ujac).__mul__(v)
            raise ValueError("Dimension mismatch")
        elif np.isscalar(v):
            return ADI(u.val*v, [v*ju for ju in u.jac])
        elif isinstance(v, np.ndarray):
            if len(u) == 1:
                val = u.val * v
                jac = [sps.diags(v.flat,0)*sps.bmat([[j]]*len(v)) for j in u.jac]
                return ADI(val, jac)
            if len(v) == 1:
                return ADI(u.val*v, [v*ju for ju in u.jac])
            if len(u.val) == len(v):
                v = np.atleast_2d(v)
                vJu = [sps.diags(v.flat, 0)*ju for ju in u.jac] # MATRIX multiplication
                return ADI(u.val*v, vJu)
            raise ValueError("Dimension mismatch")
        else:
            raise ValueError("u*v: v must be scalar or ndarray.")

    def __rmul__(v, u):
        # u * v = v * u
        return v.__mul__(u)

    # rmul

    def dot(u, A): # u x A
        """Matrix multiplication, u x A."""
        if not isinstance(A, ADI):
            A = np.atleast_2d(A)
        if len(u) == 1 and A.shape[0] == 1 and A.shape[1] == 1:
            return u*A
        raise ValueError("dot(u_ad,v) only valid for 1x1 parameters.")

    def __pow__(u, v):
        return u._pow(u, v)

    # This method is static so that it can be called with non-ADI u
    # E.g. when calculating 2**u, where u is ADI.
    @staticmethod
    def _pow(u, v):
        """Elementwise power, u**v."""
        if not isinstance(v, ADI): # u is AD, v is a scalar or vector
            v = np.atleast_2d(v)
            tmp = v*u.val**(v-1)
            uvJac = [sps.diags(tmp.flat, 0)*ju for ju in u.jac]
            return ADI(u.val**v, uvJac)
        elif not isinstance(u, ADI): # u is a scalar, v is AD
            tmp = u**v.val*np.log(u)
            uvJac = [sps.diags(tmp.flat, 0)*jv for jv in v.jac]
            return ADI(u**v.val, uvJac)
        else: # u and v are ADI objects of same length
            if len(u) != len(v):
                raise ValueError("Must be same length")
            # d(u^v)/dx = diag(u^v o (v / u))*
            # + diag(u^v o log(u))*J
            tmp1 = u.val**v.val * v.val/u.val
            tmp2 = u.val**v.val * np.log(u.val)
            uvJacPart1 = [sps.diags(tmp1.flat, 0)*ju for ju in u.jac]
            uvJacPart2 = [sps.diags(tmp2.flat, 0)*jv for jv in v.jac]
            uvJac = [a+b for (a,b) in zip(uvJacPart1, uvJacPart2)]
            return ADI(u.val**v.val, uvJac)

    def __rpow__(v, u):
        """u**v where u is not ADI."""
        return v._pow(u, v)

    def __div__(u, v):
        return u.__truediv__(v)

    def __truediv__(u, v):
        return u * v**(-1.0)

    def __rdiv__(v, u):
        return v.__rtruediv__(u)

    def __rtruediv__(v, u):
        return u * v**(-1.0)


    # div /truediv

    # getattr

    # setattr

    # max

    # min

    # sum

    # cumsum



def initVariablesADI(*variables):
    # Convert all inputs to arrays
    variables = map(np.atleast_2d, variables)
    numvals = np.array([len(variable) for variable in variables])
    n = len(variables)

    ret = [None]*n
    for i in range(n):
        nrows = numvals[i]
        # Set Jacobians wrt other variables to zero-matrices
        jac = [None]*n
        for j in np.r_[0:i, (i+1):n]:
            ncols = numvals[j]
            jac[j] = scipy.sparse.csr_matrix((nrows, ncols))

        # Set Jacobian of current variable wrt itself to the identity matrix.
        jac[i] = scipy.sparse.identity(nrows, format="csr")

        ret[i] = ADI(variables[i], jac)
    return ret


def _lMultDiag(d, J1):
    """TODO"""
    print("d", d)
    print("J1", J1)
    n = len(d)
    if np.any(d):
        ix = np.arange(n)
        #csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        D = csr_matrix((d.ravel(), (ix, ix)), shape=(n,n))
    else:
        D = 0

    # J1 is a list of sparse blocks comprising the Jacobian
    J = [None] * len(J1)
    for k in range(len(J)):
        J[k] = D*J1[k]
    return J
