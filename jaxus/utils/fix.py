import numpy as np


def fix_extent(extent):
    """Ensures that the extent is in the form (x0, x1, z0, z1)"""
    e0, e1, e2, e3 = extent
    return np.array([min(e0, e1), max(e0, e1), min(e2, e3), max(e2, e3)])
