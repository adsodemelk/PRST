"""
Functions for setting simulation parameters.
"""
import numpy as np
gravity = np.array([0, 0, 9.80665])
def gravity_reset():
    """Resets gravity to default value."""
    global gravity
    gravity = np.array([0, 0, 9.80665])
