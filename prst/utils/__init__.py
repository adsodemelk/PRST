__all__ = ["rldecode", "rlencode", "units", "mcolon"]

import numpy as np

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
