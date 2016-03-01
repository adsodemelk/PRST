from __future__ import print_function

__all__ = ["rldecode", "rlencode", "units", "mcolon", "recursive_diff"]

import numpy as np

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
