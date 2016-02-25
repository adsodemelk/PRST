import os

def getpath(filename):
    """
    Get the absolute path to a file in this directory.

    Example:
        > getpath("matrix.mat")
        > "/home/johndoe/prst/tests/matrix.mat"


    Helpful for running the tests from any location.
    """
    tests_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(tests_dir, filename))
