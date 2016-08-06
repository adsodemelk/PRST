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
            Not compatible with matrices (neither np.matrix nor
            scipy.sparse.spmatrix).

        jacobian(list[scipy.sparse.csr_matrix]):
            The Jacobian of the object. Split into parts to improve
            performance.

    Comment:
        This class is typically instantiated for a set of variables using
        initVariablesADI, not by itself.

        Many methods found in `np.ndarray` are also implemented by ADI. Example:

            x, = initVariablesADI(np.array([[2, 3, 4]]).T)
            y = x.log()
            z = x.sum()

        Using "np." methods is not supported yet, e.g., `np.dot(A, x)` where x
        is an ADI object will not work as expected, and is not recommended. A
        compatability layer, `prst.utils.npad` is provided. `npad.dot(A, x)`
        will work correctly for any number of AD arguments, and uses `np.dot(A,
        x)` if neither arguments are AD objects. Future versions of NumPy
        (>0.12) will most likely deprecate `npad` with the __numpy_ufunc__
        functionality.

    See also:
        initVariablesADI
    """
    # Requires __numpy_ufunc__ for syntactical sugar. Hopefully will be added to NumPy 1.12...
    # https://github.com/numpy/numpy/issues/7519

    __array_priority__ = 10000
    ndim = 2

    def __init__(self, val, jac):
        self.val = val
        self.jac = jac
        if not isinstance(self.jac, list):
            self.jac = [self.jac,]

    def __repr__(self):
        jacstring = str([block.shape for block in self.jac])
        return "(val: {0}.T, jac block sizes: {1})".format(self.val.T, jacstring)

    def pprint(self, name=None):
        """
        Pretty-print full matrices with limited decimals.

        Example:

            import numpy as np
            from prst.utils import initVariablesADI

            x0 = np.array([[1,2,3,2,3]]).T
            x, = initVariablesADI(x0)
            y = x**2
            y.pprint()

        Output:

            ADI properties
                val: [[1 4 9 4 9]].T

                jac[0]  [[ 2.  0.  0.  0.  0.]
                         [ 0.  4.  0.  0.  0.]
                         [ 0.  0.  6.  0.  0.]
                         [ 0.  0.  0.  4.  0.]
                         [ 0.  0.  0.  0.  6.]]
        """
        namestr = ""
        if name:
            namestr = name + " "
        lines = [
            namestr + "ADI properties",
            "\tval: " + str(self.val.T) + ".T",
        ]
        for i, j in enumerate(self.jac):
            lines.append("\n\tjac[" + str(i) + "]" + "\t" + str(j.toarray()).replace("\n", "\n\t\t"))
        lines.append("")
        print("\n".join(lines))

    def copy(self):
        return copy.deepcopy(self)

    #def __len__(self):
        #raise NotImplementedError("Use shape[0]. See http://stackoverflow.com/questions/37529715/")

    @property
    def shape(self):
        return self.val.shape

    def __ge__(u, v):
        try:
            return u.val >= v.val
        except AttributeError:
            return u.val >= v

    def __gt__(u, v):
        try:
            return u.val > v.val
        except AttributeError:
            return u.val > v

    def __le__(u, v):
        try:
            return u.val <= v.val
        except AttributeError:
            return u.val <= v

    def __lt__(u, v):
        try:
            return u.val < v.val
        except AttributeError:
            return u.val < v

    def __pos__(u): # +u
        return u.copy()

    def __neg__(u): # -u
        return ADI(-u.val, [-j for j in u.jac])

    def __add__(u, v): # u + v
        if isinstance(v, ADI):
            if u.val.shape[0] == v.val.shape[0]:
                return ADI(u.val + v.val, [ju+jv for (ju,jv) in zip(u.jac, v.jac)])
            if v.val.shape[0] == 1:
                # Tile v.jac to same length as u.jac since sparse matrices
                # don't broadcast properly.
                # https://github.com/scipy/scipy/issues/2128
                vjac = [sps.bmat([[j]]*len(u.val)) for j in v.jac]
                retjac = [ju+jv for (ju,jv) in zip(u.jac, vjac)]
                return ADI(u.val+v.val, retjac)
            if u.val.shape[0] == 1:
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
        else:
            v = np.atleast_2d(v)
            if len(u.val) == 1:
                val = u.val * v
                jac = [sps.diags(v.flat,0)*sps.bmat([[j]]*len(v)) for j in u.jac]
                return ADI(val, jac)
            if len(v) == 1:
                return ADI(u.val*v, [v.flat[0]*ju for ju in u.jac])
            if len(u.val) == len(v):
                vJu = [sps.diags(v.flat, 0)*ju for ju in u.jac] # MATRIX multiplication
                return ADI(u.val*v, vJu)
            raise ValueError("Dimension mismatch")

    def __rmul__(v, u):
        # u * v = v * u
        return v.__mul__(u)

    def dot(u, A): # u x A
        return _dot(u, A)

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
            uvJac = [_spdiag(tmp)*ju for ju in u.jac]
            return ADI(u.val**v, uvJac)
        elif not isinstance(u, ADI): # u is a scalar, v is AD
            u = np.atleast_2d(u)
            tmp = u**v.val*np.log(u)
            uvJac = [sps.diags(tmp.flat, 0)*jv for jv in v.jac]
            return ADI(u**v.val, uvJac)
        else: # u and v are ADI objects of same length
            if  len(u.val) != len(v.val):
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
        raise DeprecationWarning("Add 'from __future__ import division'.")

    def __truediv__(u, v):
        return u * v**(-1.0)

    def __rdiv__(v, u):
        raise DeprecationWarning("Add 'from __future__ import division'.")

    def __rtruediv__(v, u):
        return u * v**(-1.0)

    def __getitem__(u, s):
        """
        Slices the column array using NumPy syntax.

        Examples: (x is ADI object)

            x[(2,1),:]
            x[1]
            x[1,:]
            x[np.array([True,True,False])]
            x[np.array([False,False,False]),:]
            x[np.array([2,1,0]),:]
            x[np.array([2]),:]
            x[::-1]
        """
        val = np.atleast_2d(u.val[s])
        if val.shape[0] != 1 and val.shape[1] != 1:
            raise ValueError("Slice type not supported")
        if val.shape[1] != 1:
            val = val.T
        try:
            s = s[0]
        except TypeError:
            pass
        jac = [j[s,:] for j in u.jac]
        return ADI(val, jac)

    def __setitem__(u, s, v):
        """
        Sets values in ADI vector.

        If the right side is non-ADI, the corresponding Jacobian rows are set to zero.
        If the right side is ADI, the corresponding Jacobian rows are overwritten.
        """
        if isinstance(v, ADI):
            # This part is not so pretty, and could probably
            # be improved.
            if u.val[s].ndim <= 1:
                u.val[s] = v.val.ravel()
            elif u.val[s].ndim == 2:
                u.val[s] = v.val
            else:
                raise ValueError("This should never happen.")
            try:
                s = s[0]
            except TypeError:
                pass
            for i in range(len(u.jac)):
                u.jac[i][s] = v.jac[i]
        else:
            u.val[s] = v
            try:
                s = s[0]
            except TypeError:
                pass
            for i in range(len(u.jac)):
                u.jac[i][s] = 0

    def max(u):
        """Return the maximum element in the array."""
        i = np.argmax(u.val)
        return ADI(np.atleast_2d(u.val[i,:]), [j[i,:] for j in u.jac])

    def min(u):
        """Return the minimum element in the array."""
        i = np.argmin(u.val)
        return ADI(np.atleast_2d(u.val[i,:]), [j[i,:] for j in u.jac])

    def sum(u):
        """Return the sum of the array elements."""
        val = u.val.sum(keepdims=True)
        jac = [sps.csr_matrix(j.sum(axis=0)) for j in u.jac]
        return ADI(val, jac)

    def sin(u):
        """Return element-wise sine of array."""
        val = np.sin(u.val)
        cosval = np.cos(u.val)
        jac = [sps.diags(cosval.flat, 0)*j for j in u.jac]
        return ADI(val, jac)

    def cos(u):
        """Return element-wise cosine of array."""
        val = np.cos(u.val)
        msinval = -np.sin(u.val)
        jac = [sps.diags(msinval.flat, 0)*j for j in u.jac]
        return ADI(val, jac)

    def exp(u):
        val = np.exp(u.val)
        jac = [sps.diags(val.flat, 0)*j for j in u.jac]
        return ADI(val, jac)

    def log(u):
        val = np.log(u.val)
        m = sps.diags((1/u.val).flat, 0)
        jac = [m*j for j in u.jac]
        return ADI(val, jac)

    def sign(u):
        return np.sign(u.val)

    def abs(u):
        val = np.abs(u.val)
        sgn = np.sign(u.val)
        jac = [sps.diags(sgn.flat, 0)*j for j in u.jac]
        return ADI(val, jac)

    def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
        """Placeholder method for future NumPy versions."""
        raise NotImplementedError("NumPy has finally added __numpy_ufunc__ support, but "
                                   "PRST has not added support yet.")

# NumPy binary ufunc wrappers
def _dot(u, v):
    """Matrix multiplication."""
    if isinstance(u, ADI) and isinstance(v, ADI):
        # u_ad, v_ad
        assert u.val.shape[0] == v.val.shape[0] == 1, "dot(ad,ad) only valid for 1x1 arguments"
        return u * v
    elif isinstance(u, ADI) and not isinstance(v, ADI):
        # u_ad, v
        v = np.atleast_2d(v)
        assert v.shape[0] == 1, "dot(ad,vec) only valid for 1x1 vec."
        return u*v
    elif not isinstance(u, ADI) and isinstance(v, ADI):
        # u, v_ad
        if not hasattr(u, "dot"):
            u = np.atleast_2d(u)
        u_sp = sps.csr_matrix(u)
        return ADI(u.dot(v.val), [u_sp*j for j in v.jac])
    else:
        # u, v
        if hasattr(u, "dot"):
            return u.dot(v)
        return np.dot(u, v)

def _tile(A, reps):
    if isinstance(A, ADI):
        if len(reps) != 2 or reps[1] != 1:
            raise TypeError("AD vectors can only be tiled vertically.")
        val = np.tile(A.val, reps)
        jac = [sps.bmat([[j]]*reps[0]) for j in A.jac]
        return ADI(val, jac)
    else:
        return np.tile(A, reps)

# Numpy unary ufunc wrappers
# The unary wrappers are all following the same formula, and can possibly be
# removed entirely by making `npad` more magic with __getattr__.
def _sign(u):
    if isinstance(u, ADI):
        return u.sign()
    else:
        return np.sign(u)

def _abs(u):
    """np.abs for AD array."""
    if isinstance(u, ADI):
        return u.abs()
    else:
        return np.abs(u)

def _exp(u):
    """np.exp for AD array."""
    if isinstance(u, ADI):
        return u.exp()
    else:
        return np.abs(u)

# NumPy n-ary functions

def _vstack(tup):
    """np.vstack for AD array."""
    vals = np.vstack((u.val for u in tup))
    jacs = []
    num_jacs = len(tup[0].jac)
    for j in range(num_jacs):
        jacs.append(sps.bmat([[u.jac[j]] for u in tup]))
    return ADI(vals, jacs)

def _concatenate(tup, axis):
    """np.concatenate for AD array."""
    if axis != 0:
        raise TypeError("ADI objects can only be concatenated vertically.")
    return _vstack(tup)

# Register ufunc wrappers so they can be easily imported.
npad = Struct()
# n-ary
npad.vstack = _vstack
npad.concatenate = _concatenate
# binary
npad.dot = _dot
npad.tile = _tile
# unary
npad.sign = _sign
npad.abs = _abs

def initVariablesADI(*variables):
    """
    Returns AD (automatic differentiation) variables.

    See `help(prst.utils.ADI)` for documentation.
    """
    # Convert all inputs to column arrays
    vals = list(variables)
    for i in range(len(vals)):
        vals[i] = np.atleast_2d(vals[i])
        if vals[i].shape[1] == 0:
            vals[i] = vals[i].reshape(-1,1)
        elif vals[i].shape[1] != 1:
            raise ValueError("AD variables must be column vectors.")

    numvals = np.array([len(val) for val in vals])
    n = len(vals)

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

        ret[i] = ADI(vals[i], jac)
    return ret

def _spdiag(val_column):
    """Improved version of scipy.sparse.diags."""
    if val_column.shape[0] == 0:
        return sps.csr_matrix((1,0))
    return sps.diags(val_column.flat, 0, format="csr")
