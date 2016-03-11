"""
Python Reservoir Simulation Toolbox.




"""
from __future__ import print_function
import sys

import numpy as np

__all__ = ["gridprocessing", "io", "incomp", "plotting", "utils", "params"]

import logging
log = logging.getLogger('prst')
log.setLevel(logging.DEBUG)

def warning(*args):
    print("WARNING: ", *args, file=sys.stderr)

verbosity = False
def verbosity_set(value):
    """Sets PRST verbosity."""
    assert value in (False, True, 0, 1, 2), "Invalid verbosity level"
    global verbosity
    verbosity = value

def verbosity_reset():
    """Resets PRST verbosity."""
    global verbosity
    verbosity = False

gravity = np.array([0, 0, 9.80665])
def gravity_reset():
    """Resets gravity to default value."""
    global gravity
    gravity = np.array([0, 0, 9.80665])

import __builtin__
try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile
