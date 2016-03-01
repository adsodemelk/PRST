"""
Functions for setting simulation parameters.
"""
__all__ = ["rock", "wells_and_bc"]
from . import rock, wells_and_bc

import numpy as np
gravity = np.array([0, 0, 9.80665])
def gravity_reset():
    """Resets gravity to default value."""
    global gravity
    gravity = np.array([0, 0, 9.80665])
